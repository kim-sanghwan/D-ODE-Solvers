import os
import logging
import time
import glob
from tkinter import E

import blobfile as bf

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from models.diffusion import Model
from models.improved_ddpm.unet import UNetModel as ImprovedDDPM_Model
from models.guided_diffusion.unet import UNetModel as GuidedDiffusion_Model
from models.guided_diffusion.unet import EncoderUNetModel as GuidedDiffusion_Classifier
from models.guided_diffusion.unet import SuperResModel as GuidedDiffusion_SRModel
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from evaluate.fid_score import calculate_fid_given_paths

import torchvision.utils as tvu


def load_data_for_worker(base_samples, batch_size, cond_class):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if cond_class:
            label_arr = obj["arr_1"]
    buffer = []
    label_buffer = []
    while True:
        for i in range(len(image_arr)):
            buffer.append(image_arr[i])
            if cond_class:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = torch.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if cond_class:
                    res["y"] = torch.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == 'cosine':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, rank=None):
        self.args = args
        self.config = config
        if rank is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            device = rank
            self.rank = rank
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        self.alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_cumprod_prev = torch.concat(
            [torch.ones(1).to(device), self.alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.concat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        self.lambdas = None
        self.variance_alpha_sequence = None
        self.mean_alpha_sequence = None
        
    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.concat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def sample(self):
        if self.config.model.model_type == 'improved_ddpm':
            model = ImprovedDDPM_Model(
                in_channels=self.config.model.in_channels,
                model_channels=self.config.model.model_channels,
                out_channels=self.config.model.out_channels,
                num_res_blocks=self.config.model.num_res_blocks,
                attention_resolutions=self.config.model.attention_resolutions,
                dropout=self.config.model.dropout,
                channel_mult=self.config.model.channel_mult,
                conv_resample=self.config.model.conv_resample,
                dims=self.config.model.dims,
                use_checkpoint=self.config.model.use_checkpoint,
                num_heads=self.config.model.num_heads,
                num_heads_upsample=self.config.model.num_heads_upsample,
                use_scale_shift_norm=self.config.model.use_scale_shift_norm
            )
        elif self.config.model.model_type == "guided_diffusion":
            if self.config.model.is_upsampling:
                model = GuidedDiffusion_SRModel(
                    image_size=self.config.model.large_size,
                    in_channels=self.config.model.in_channels,
                    model_channels=self.config.model.model_channels,
                    out_channels=self.config.model.out_channels,
                    num_res_blocks=self.config.model.num_res_blocks,
                    attention_resolutions=self.config.model.attention_resolutions,
                    dropout=self.config.model.dropout,
                    channel_mult=self.config.model.channel_mult,
                    conv_resample=self.config.model.conv_resample,
                    dims=self.config.model.dims,
                    num_classes=self.config.model.num_classes,
                    use_checkpoint=self.config.model.use_checkpoint,
                    use_fp16=self.config.model.use_fp16,
                    num_heads=self.config.model.num_heads,
                    num_head_channels=self.config.model.num_head_channels,
                    num_heads_upsample=self.config.model.num_heads_upsample,
                    use_scale_shift_norm=self.config.model.use_scale_shift_norm,
                    resblock_updown=self.config.model.resblock_updown,
                    use_new_attention_order=self.config.model.use_new_attention_order,
                )
            else:
                model = GuidedDiffusion_Model(
                    image_size=self.config.model.image_size,
                    in_channels=self.config.model.in_channels,
                    model_channels=self.config.model.model_channels,
                    out_channels=self.config.model.out_channels,
                    num_res_blocks=self.config.model.num_res_blocks,
                    attention_resolutions=self.config.model.attention_resolutions,
                    dropout=self.config.model.dropout,
                    channel_mult=self.config.model.channel_mult,
                    conv_resample=self.config.model.conv_resample,
                    dims=self.config.model.dims,
                    num_classes=self.config.model.num_classes,
                    use_checkpoint=self.config.model.use_checkpoint,
                    use_fp16=self.config.model.use_fp16,
                    num_heads=self.config.model.num_heads,
                    num_head_channels=self.config.model.num_head_channels,
                    num_heads_upsample=self.config.model.num_heads_upsample,
                    use_scale_shift_norm=self.config.model.use_scale_shift_norm,
                    resblock_updown=self.config.model.resblock_updown,
                    use_new_attention_order=self.config.model.use_new_attention_order,
                )
        else:
            model = Model(self.config)

        model = model.to(self.rank)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}

        if "ckpt_dir" in self.config.model.__dict__.keys():
            ckpt_dir = os.path.expanduser(self.config.model.ckpt_dir)
            states = torch.load(
                ckpt_dir,
                map_location=map_location
            )
            # states = {f"module.{k}":v for k, v in states.items()}
            if self.config.model.model_type == 'improved_ddpm' or self.config.model.model_type == 'guided_diffusion':
                model.load_state_dict(states, strict=True)
                if self.config.model.use_fp16:
                    model.convert_to_fp16()
            else:
                # TODO: FIXME
                # model.load_state_dict(states[0], strict=True)
                if self.config.data.dataset  == "CELEBA":
                    torch.cuda.set_device('cuda:0')
                    model = torch.nn.DataParallel(model).cuda()
                else:
                    model.load_state_dict(states, strict=True)

            if self.config.model.ema: # for celeba 64x64 in DDIM
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None

            if self.config.sampling.cond_class and not self.config.model.is_upsampling:
                classifier = GuidedDiffusion_Classifier(
                    image_size=self.config.classifier.image_size,
                    in_channels=self.config.classifier.in_channels,
                    model_channels=self.config.classifier.model_channels,
                    out_channels=self.config.classifier.out_channels,
                    num_res_blocks=self.config.classifier.num_res_blocks,
                    attention_resolutions=self.config.classifier.attention_resolutions,
                    channel_mult=self.config.classifier.channel_mult,
                    use_fp16=self.config.classifier.use_fp16,
                    num_head_channels=self.config.classifier.num_head_channels,
                    use_scale_shift_norm=self.config.classifier.use_scale_shift_norm,
                    resblock_updown=self.config.classifier.resblock_updown,
                    pool=self.config.classifier.pool
                )
                ckpt_dir = os.path.expanduser(self.config.classifier.ckpt_dir)
                states = torch.load(
                    ckpt_dir,
                    map_location=map_location,
                )
                # states = {f"module.{k}":v for k, v in states.items()}
                classifier = classifier.to(self.rank)
                # classifier = DDP(classifier, device_ids=[self.rank])
                classifier.load_state_dict(states, strict=True)
                if self.config.classifier.use_fp16:
                    classifier.convert_to_fp16()
                    # classifier.module.convert_to_fp16()
            else:
                classifier = None
        else:
            classifier = None
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}", root="/cluster/work/cvl/tangha/sanghwan/ddpm_ckpt/",)
            if self.rank == 0:
                print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=map_location))

        model.eval()

        if self.args.fid:
            if not os.path.exists(os.path.join(self.args.exp, "fid.npy")):
                self.sample_fid(model, classifier=classifier)
                torch.distributed.barrier()
                if self.rank == 0:
                    print("Begin to compute FID...")
                    fid = calculate_fid_given_paths((self.config.sampling.fid_stats_dir, self.args.image_folder), batch_size=self.config.sampling.fid_batch_size, device=self.device, dims=2048, num_workers=8)
                    print("FID: {}".format(fid))
                    np.save(os.path.join(self.args.exp, "fid"), fid)
        # elif self.args.interpolation:
        #     self.sample_interpolation(model)
        # elif self.args.sequence:
        #     self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model, classifier=None):
        config = self.config
        total_n_samples = config.sampling.fid_total_samples
        world_size = torch.cuda.device_count()
        if total_n_samples % config.sampling.batch_size != 0:
            raise ValueError("Total samples for sampling must be divided exactly by config.sampling.batch_size, but got {} and {}".format(total_n_samples, config.sampling.batch_size))
        if len(glob.glob(f"{self.args.image_folder}/*.png")) == total_n_samples:
            return
        else:
            n_rounds = total_n_samples // config.sampling.batch_size // world_size
        img_id = self.rank * total_n_samples // world_size

        if self.config.model.is_upsampling:
            base_samples_total = load_data_for_worker(self.args.base_samples, config.sampling.batch_size, config.sampling.cond_class)

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                # torch.cuda.synchronize()
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()

                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                if self.config.model.is_upsampling:
                    base_samples = next(base_samples_total)
                else:
                    base_samples = None

                x, classes = self.sample_image(x, model, classifier=classifier, base_samples=base_samples)

                # end.record()
                # torch.cuda.synchronize()
                # t_list.append(start.elapsed_time(end))
                x = inverse_data_transform(config, x)
                for i in range(x.shape[0]):
                    if classes is None:
                        path = os.path.join(self.args.image_folder, f"{img_id}.png")
                    else:
                        path = os.path.join(self.args.image_folder, f"{img_id}_{int(classes.cpu()[i])}.png")
                    tvu.save_image(x.cpu()[i], path)
                    img_id += 1
        # # Remove the time evaluation of the first batch, because it contains extra initializations
        # print('time / batch', np.mean(t_list[1:]) / 1000., 'std', np.std(t_list[1:]) / 1000.)

    def sample_sequence(self, model, classifier=None):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False, classifier=classifier)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.concat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.concat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True, classifier=None, base_samples=None):
        assert last
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        classifier_scale = self.config.sampling.classifier_scale if self.args.scale is None else self.args.scale
        if self.config.sampling.cond_class:
            if self.args.fixed_class is None:
                classes = torch.randint(low=0, high=self.config.data.num_classes, size=(x.shape[0],)).to(x.device)
            else:
                classes = torch.randint(low=self.args.fixed_class, high=self.args.fixed_class + 1, size=(x.shape[0],)).to(x.device)
        else:
            classes = None
        
        if base_samples is None:
            if classes is None:
                model_kwargs = {}
            else:
                model_kwargs = {"y": classes}
        else:
            model_kwargs = {"y": base_samples["y"], "low_res": base_samples["low_res"]}

        def model_fn(x, t, **model_kwargs):
            out = model(x, t, **model_kwargs)
            if "out_channels" in self.config.model.__dict__.keys():
                if self.config.model.out_channels == 6:
                    return torch.split(out, 3, dim=1)[0]
            return out

        # for DEIS and PNDM
        def eps_fn(x, s_t, **model_kwargs):
            vec_t = (torch.ones(x.shape[0])).float().to(self.device) * s_t
            with torch.no_grad():
                # ! the checkpoint need vec_t shift 1 :(
                out = model(x, vec_t - 1, **model_kwargs)
            if "out_channels" in self.config.model.__dict__.keys():
                if self.config.model.out_channels == 6:
                    return torch.split(out, 3, dim=1)[0]
            return out


        if self.args.sample_type == "ddpm":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "uniform_torch":
                temp_seq = torch.linspace(0, self.num_timesteps, self.args.timesteps+1)
                seq = [int(s.item()) for s in temp_seq[:self.args.timesteps]]
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps
            xs, _ = ddpm_steps(x, seq, model_fn, self.betas, classifier=classifier, is_cond_classifier=self.config.sampling.cond_class, classifier_scale=classifier_scale, **model_kwargs)
            x = xs[-1]

        elif self.args.sample_type == "ddim":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
                seq = seq[:self.args.timesteps]
            elif self.args.skip_type == "uniform_torch":
                temp_seq = torch.linspace(0, self.num_timesteps, self.args.timesteps+1)
                seq = [int(s.item()) for s in temp_seq[:self.args.timesteps]]
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddim_steps
            xs, _ = ddim_steps(x, seq, model_fn, self.betas, eta=self.args.eta, classifier=classifier, is_cond_classifier=self.config.sampling.cond_class, classifier_scale=classifier_scale, **model_kwargs)
            x = xs[-1]

        elif self.args.sample_type == "d_ddim":
            SCALE = self.args.d_ode_scale
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
                seq = seq[:self.args.timesteps]
                if self.lambdas == None:
                    steps_inference = SCALE*self.args.timesteps
                    ref_skip = self.num_timesteps // steps_inference
                    teacher_seq = range(0, self.num_timesteps, ref_skip)
            elif self.args.skip_type == "uniform_torch": # proper for not divisible timesteps e.g. self.num_timesteps%self.args.timesteps!=0
                temp_seq = torch.linspace(0, self.num_timesteps, self.args.timesteps+1)
                seq = [int(s.item()) for s in temp_seq[:self.args.timesteps]]
                if self.lambdas == None:
                    # Maximum 1000 steps
                    teacher_timesteps = min( self.num_timesteps, SCALE*self.args.timesteps)
                    teacher_seq = torch.linspace(0, self.num_timesteps, teacher_timesteps+1)
                    teacher_seq = [int(s.item()) for s in teacher_seq[:teacher_timesteps]]
            else:
                raise NotImplementedError

            # perform ddim with more steps first to estimate parameters
            from functions.denoising import teacher_ddim_steps, d_ddim_get_lambdas, d_ddim_steps
            if self.lambdas == None: # At first batch, we need to obtain lambdas
                # Teacher sampling
                xs = teacher_ddim_steps(x, teacher_seq, model_fn, self.betas, eta=self.args.eta, classifier=classifier, is_cond_classifier=self.config.sampling.cond_class, classifier_scale=classifier_scale, **model_kwargs)

                # Set teacher targets
                teacher_targets = []
                temp_scale = len(teacher_seq)//self.args.timesteps
                for i in range(self.args.timesteps):
                    teacher_targets.append(xs[i*temp_scale])
                teacher_targets.append(xs[-1]) # add x0    
                teacher_targets = torch.stack(teacher_targets, dim=0)

                xs, self.lambdas = d_ddim_get_lambdas(x, teacher_targets, seq, model_fn, self.betas, eta=self.args.eta, classifier=classifier, is_cond_classifier=self.config.sampling.cond_class, classifier_scale=classifier_scale, **model_kwargs)
                x = xs[-1]
                print(f'lambdas: {self.lambdas}', flush=True)
            else: # From second batch, we reuse lambdas
                xs = d_ddim_steps(x, self.lambdas, seq, model_fn, self.betas, eta=self.args.eta, classifier=classifier, is_cond_classifier=self.config.sampling.cond_class, classifier_scale=classifier_scale, **model_kwargs)
                x = xs[-1]

        elif self.args.sample_type == "ipndm":
            from th_deis import DiscreteVPSDE, get_sampler
            vpsde = DiscreteVPSDE(self.alphas_cumprod)
            sampler_fn = get_sampler(
                vpsde,
                eps_fn,
                num_step=self.args.timesteps,
                method = "ipndm",
                #for classifier guidance
                classifier=classifier, 
                is_cond_classifier=self.config.sampling.cond_class, 
                classifier_scale=classifier_scale, 
                **model_kwargs
            )
            x = sampler_fn(x)

        elif self.args.sample_type == "d_ipndm":
            SCALE = self.args.d_ode_scale
            from th_deis import DiscreteVPSDE, d_ode_get_sampler
            vpsde = DiscreteVPSDE(self.alphas_cumprod)

            if self.lambdas == None: # At first batch, we need to obtain lambdas
                # Teacher sampling
                teacher_num_timesteps = min( self.num_timesteps, SCALE*self.args.timesteps) # Maximum 1000 steps
                
                # to prevent recursion error for teacher_steps=1000
                import sys 
                sys.setrecursionlimit(2000)    

                teacher_sampler_fn = d_ode_get_sampler(
                    vpsde,
                    eps_fn,
                    num_step=teacher_num_timesteps,
                    method = "teacher_ipndm",
                    #for classifier guidance
                    classifier=classifier, 
                    is_cond_classifier=self.config.sampling.cond_class, 
                    classifier_scale=classifier_scale, 
                    **model_kwargs
                )
                xs, _ = teacher_sampler_fn(x)

                # Set teacher targets
                teacher_targets = []
                temp_scale = teacher_num_timesteps//self.args.timesteps
                for i in range(self.args.timesteps):
                    teacher_targets.append(xs[i*temp_scale])
                teacher_targets.append(xs[-1]) # add x0
                teacher_targets = torch.stack(teacher_targets, dim=0)

                sampler_fn = d_ode_get_sampler(
                    vpsde,
                    eps_fn,
                    num_step=self.args.timesteps,
                    method = "distilled_ipndm",
                    lambdas=self.lambdas,
                    teacher_targets=teacher_targets,
                    #for classifier guidance
                    classifier=classifier, 
                    is_cond_classifier=self.config.sampling.cond_class, 
                    classifier_scale=classifier_scale, 
                    **model_kwargs
                )
                xs, x, self.lambdas = sampler_fn(x)
                print(f'lambdas: {self.lambdas}', flush=True)

            else: # From second batch, we reuse lambdas
                sampler_fn = d_ode_get_sampler(
                    vpsde,
                    eps_fn,
                    num_step=self.args.timesteps,
                    method = "distilled_ipndm",
                    lambdas=self.lambdas,
                    #for classifier guidance
                    classifier=classifier, 
                    is_cond_classifier=self.config.sampling.cond_class, 
                    classifier_scale=classifier_scale, 
                    **model_kwargs   
                )
                x = sampler_fn(x)

        elif self.args.sample_type == "dpmsolver":
            from dpm_solver.sampler import NoiseScheduleVP, model_wrapper, DPM_Solver
            def classifier_fn(x, t, y, **classifier_kwargs):
                logits = classifier(x, t)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                return log_probs[range(len(logits)), y.view(-1)]

            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
            model_fn_continuous = model_wrapper(
                model_fn,
                noise_schedule,
                model_type="noise",
                model_kwargs=model_kwargs,
                guidance_type="uncond" if classifier is None else "classifier",
                condition=model_kwargs["y"] if "y" in model_kwargs.keys() else None,
                guidance_scale=classifier_scale,
                classifier_fn=classifier_fn,
                classifier_kwargs={},
            )
            dpm_solver = DPM_Solver(
                model_fn_continuous,
                noise_schedule,
                algorithm_type=self.args.sample_type,
                correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
            )
            x = dpm_solver.sample(
                x,
                steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                order=self.args.dpm_solver_order,
                skip_type=self.args.skip_type,
                method=self.args.dpm_solver_method,
                lower_order_final=self.args.lower_order_final,
                denoise_to_zero=self.args.denoise,
                solver_type=self.args.dpm_solver_type,
                atol=self.args.dpm_solver_atol,
                rtol=self.args.dpm_solver_rtol,
            )

        elif self.args.sample_type == "d_dpmsolver":
            SCALE = self.args.d_ode_scale
            if self.args.skip_type == "time_uniform": # proper for not divisible timesteps e.g. self.num_timesteps%self.args.timesteps!=0
                temp_seq = torch.linspace(0, self.num_timesteps, self.args.timesteps+1)
                seq = [int(s.item()) for s in temp_seq[:self.args.timesteps]]
                if self.lambdas == None:
                    # Maximum 1000 steps
                    teacher_timesteps = min( self.num_timesteps, SCALE*self.args.timesteps)
                    teacher_seq = torch.linspace(0, self.num_timesteps, teacher_timesteps+1)
                    teacher_seq = [int(s.item()) for s in teacher_seq[:teacher_timesteps]]
            else:
                raise NotImplementedError

            # only for 2nd order
            from functions.denoising import teacher_ddim_steps
            from dpm_solver.d_sampler import NoiseScheduleVP, model_wrapper, D_DPM_Solver, Get_lambdas_DPM_Solver
            def classifier_fn(x, t, y, **classifier_kwargs):
                logits = classifier(x, t)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                return log_probs[range(len(logits)), y.view(-1)]

            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
            model_fn_continuous = model_wrapper(
                model_fn,
                noise_schedule,
                model_type="noise",
                model_kwargs=model_kwargs,
                guidance_type="uncond" if classifier is None else "classifier",
                condition=model_kwargs["y"] if "y" in model_kwargs.keys() else None,
                guidance_scale=classifier_scale,
                classifier_fn=classifier_fn,
                classifier_kwargs={},
            )

            if self.lambdas == None:
                get_lambdas_solver = Get_lambdas_DPM_Solver(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.sample_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None
                )
                xs = teacher_ddim_steps(x, teacher_seq, model_fn, self.betas, eta=self.args.eta, classifier=classifier, is_cond_classifier=self.config.sampling.cond_class, classifier_scale=classifier_scale, **model_kwargs)

                # Set teacher targets
                teacher_targets = []
                temp_scale = len(teacher_seq)//self.args.timesteps
                if self.args.dpm_solver_order == 2:
                    for i in range(self.args.timesteps//2):
                        x1 = xs[(2*i+1)*temp_scale]
                        x2 = xs[(2*i+2)*temp_scale]
                        teacher_targets.append((x1.unsqueeze(0), x2.unsqueeze(0))) 
                    if self.args.timesteps % 2 != 0:
                        teacher_targets.append((xs[-1].unsqueeze(0))) 

                elif self.args.dpm_solver_order == 3: 
                    K = self.args.timesteps // 3 + 1 
                    if self.args.timesteps % 3 == 0:
                        for i in range(K-2):
                            x1 = xs[(3*i+1)*temp_scale]
                            x2 = xs[(3*i+2)*temp_scale]
                            x3 = xs[(3*i+3)*temp_scale]
                            teacher_targets.append( (x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0)) ) 
                        teacher_targets.append( (xs[-2*temp_scale-1].unsqueeze(0), xs[-temp_scale-1].unsqueeze(0)) ) 
                        teacher_targets.append( (xs[-1].unsqueeze(0)) ) 
                    elif self.args.timesteps % 3 == 1:
                        for i in range(K-1):
                            x1 = xs[(3*i+1)*temp_scale]
                            x2 = xs[(3*i+2)*temp_scale]
                            x3 = xs[(3*i+3)*temp_scale]
                            teacher_targets.append( (x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0)) ) 
                        teacher_targets.append( (xs[-1].unsqueeze(0)) ) 
                    else:
                        for i in range(K-1):
                            x1 = xs[(3*i+1)*temp_scale]
                            x2 = xs[(3*i+2)*temp_scale]
                            x3 = xs[(3*i+3)*temp_scale]
                            teacher_targets.append( (x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0)) ) 
                        teacher_targets.append( (xs[-temp_scale-1].unsqueeze(0), xs[-1].unsqueeze(0)) ) 
                else:
                    raise NotImplementedError

                x, self.lambdas = get_lambdas_solver.get_lambdas(
                    x, teacher_targets,
                    steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                    order=self.args.dpm_solver_order,
                    skip_type=self.args.skip_type,
                    method=self.args.dpm_solver_method,
                    lower_order_final=self.args.lower_order_final,
                    denoise_to_zero=self.args.denoise,
                    solver_type=self.args.dpm_solver_type,
                    atol=self.args.dpm_solver_atol,
                    rtol=self.args.dpm_solver_rtol,
                )
                print(f'lambdas: {self.lambdas}', flush=True)

            else: 
                d_dpm_solver = D_DPM_Solver(
                    model_fn_continuous,
                    noise_schedule,
                    algorithm_type=self.args.sample_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    lambdas=self.lambdas
                )
                x = d_dpm_solver.sample(
                    x,
                    steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                    order=self.args.dpm_solver_order,
                    skip_type=self.args.skip_type,
                    method=self.args.dpm_solver_method,
                    lower_order_final=self.args.lower_order_final,
                    denoise_to_zero=self.args.denoise,
                    solver_type=self.args.dpm_solver_type,
                    atol=self.args.dpm_solver_atol,
                    rtol=self.args.dpm_solver_rtol,
                )

        elif self.args.sample_type == "tab_deis":
            from th_deis import DiscreteVPSDE, get_sampler
            vpsde = DiscreteVPSDE(self.alphas_cumprod)
            sampler_fn = get_sampler(
                # args for diffusion model
                vpsde,
                eps_fn,
                num_step=self.args.timesteps,
                # args for timestamps scheduling
                ts_phase="t", # support "rho", "t", "log"
                ts_order=1.0,
                # deis choice
                method = "t_ab", # deis sampling algorithms: support "rho_rk", "rho_ab", "t_ab", "ipndm"
                ab_order= self.args.deis_order, # for "rho_ab", "t_ab" algorithms, other algorithms will ignore the arg
                # rk_method="3kutta" # for "rho_rk" algorithms, other algorithms will ignore the arg
                rk_method="2heun", # for "rho_rk" algorithms, other algorithms will ignore the arg
                #for classifier guidance
                classifier=classifier, 
                is_cond_classifier=self.config.sampling.cond_class, 
                classifier_scale=classifier_scale, 
                **model_kwargs
            )
            x = sampler_fn(x)

        elif self.args.sample_type == "d_tab_deis":
            SCALE = self.args.d_ode_scale
            from th_deis import DiscreteVPSDE, d_ode_get_sampler
            vpsde = DiscreteVPSDE(self.alphas_cumprod)

            if self.lambdas == None: # At first batch, we need to obtain lambdas
                # Teacher sampling
                teacher_num_timesteps = min( self.num_timesteps, SCALE*self.args.timesteps) # Maximum 1000 steps

                teacher_sampler_fn = d_ode_get_sampler(
                    vpsde,
                    eps_fn,
                    num_step=teacher_num_timesteps,
                    ts_phase="t", 
                    ts_order=1.0,
                    method = "teacher_t_ab",
                    ab_order= self.args.deis_order,
                    #for classifier guidance
                    classifier=classifier, 
                    is_cond_classifier=self.config.sampling.cond_class, 
                    classifier_scale=classifier_scale, 
                    **model_kwargs                    
                )
                xs, _ = teacher_sampler_fn(x)

                # Set teacher targets
                teacher_targets = []
                temp_scale = teacher_num_timesteps//self.args.timesteps
                for i in range(self.args.timesteps):
                    teacher_targets.append(xs[i*temp_scale])
                teacher_targets.append(xs[-1]) # add x0
                teacher_targets = torch.stack(teacher_targets, dim=0)

                sampler_fn = d_ode_get_sampler(
                    vpsde,
                    eps_fn,
                    num_step=self.args.timesteps,
                    ts_phase="t", 
                    ts_order=1.0,
                    method = "distilled_t_ab",
                    ab_order= self.args.deis_order,
                    lambdas=self.lambdas,
                    teacher_targets=teacher_targets,
                    #for classifier guidance
                    classifier=classifier, 
                    is_cond_classifier=self.config.sampling.cond_class, 
                    classifier_scale=classifier_scale, 
                    **model_kwargs  
                )
                xs, x, self.lambdas = sampler_fn(x)
                print(f'lambdas: {self.lambdas}', flush=True)

            else: # From second batch, we reuse lambdas
                sampler_fn = d_ode_get_sampler(
                    vpsde,
                    eps_fn,
                    num_step=self.args.timesteps,
                    ts_phase="t", 
                    ts_order=1.0,
                    method = "distilled_t_ab",
                    ab_order= self.args.deis_order,
                    lambdas=self.lambdas,
                    #for classifier guidance
                    classifier=classifier, 
                    is_cond_classifier=self.config.sampling.cond_class, 
                    classifier_scale=classifier_scale, 
                    **model_kwargs     
                )
                x = sampler_fn(x)

        else:
            raise NotImplementedError
        return x, classes
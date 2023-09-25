import torch
import torch.nn.functional as F
import math


class NoiseScheduleVP:
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=torch.float32,
        ):

        if schedule not in ['discrete', 'linear']:
            raise ValueError("Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear'".format(schedule))

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.T = 1.
            self.log_alpha_array = self.numerical_clip_alpha(log_alphas).reshape((1, -1,)).to(dtype=dtype)
            self.total_N = self.log_alpha_array.shape[1]
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
        else:
            self.T = 1.
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1

    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        """
        For some beta schedules such as cosine schedule, the log-SNR has numerical isssues. 
        We clip the log-SNR near t=T within -5.1 to ensure the stability.
        Such a trick is very useful for diffusion models with the cosine schedule, such as i-DDPM, guided-diffusion and GLIDE.
        """
        log_sigmas = 0.5 * torch.log(1. - torch.exp(2. * log_alphas))
        lambs = log_alphas - log_sigmas  
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))


def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.,
    classifier_fn=None,
    classifier_kwargs={},
):
    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * 1000.
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t * output) / sigma_t
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return alpha_t * output + sigma_t * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -sigma_t * output

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * expand_dims(sigma_t, x.dim()) * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1. or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v", "score"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


class D_DPM_Solver:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="dpmsolver++",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.,
        dynamic_thresholding_ratio=0.995,
        lambdas = None
    ):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        self.algorithm_type = algorithm_type
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val
        self.lambdas = lambdas

    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method. 
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0, t)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model. 
        """
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):

        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3,] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3,] * (K - 1) + [1]
            else:
                orders = [3,] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2,] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (K - 1) + [1]
        elif order == 1:
            K = 1
            orders = [1,] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == 'logSNR':
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[torch.cumsum(torch.tensor([0,] + orders), 0).to(device)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization. 
        """
        return self.data_prediction_fn(x, s)

    def dpm_solver_first_update(self, step, x, s, t, model_s_prev, model_s=None, return_intermediate=False):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                sigma_t / sigma_s * x
                - alpha_t * phi_1 * model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s_prev != None:
                new_model_s = model_s + self.lambdas[step][0]*(model_s-model_s_prev) 
            else:
                new_model_s = model_s + self.lambdas[step][0]*(model_s-x) 

            x_t = (
                torch.exp(log_alpha_t - log_alpha_s) * x
                - (sigma_t * phi_1) * new_model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t, model_s


    def singlestep_dpm_solver_second_update(self, step, x, s, t, model_s_prev, r1=0.5, model_s=None, return_intermediate=False, solver_type='dpmsolver'):
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_1 = torch.expm1(-h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (
                (sigma_s1 / sigma_s) * x
                - (alpha_s1 * phi_11) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    - (0.5 / r1) * (alpha_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1. / r1) * (alpha_t * (phi_1 / h + 1.)) * (model_s1 - model_s)
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_1 = torch.expm1(h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s_prev != None:
                new_model_s = model_s + self.lambdas[step][0]*(model_s-model_s_prev)
            else:
                new_model_s = model_s + self.lambdas[step][0]*(model_s-x)
            x_s1 = (
                torch.exp(log_alpha_s1 - log_alpha_s) * x
                - (sigma_s1 * phi_11) * new_model_s
            )
            model_s1 = self.model_fn(x_s1, s1)

            new_model_s1 = model_s1 + self.lambdas[step][1]*(model_s1-model_s) 
       
            if solver_type == 'dpmsolver':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (0.5 / r1) * (sigma_t * phi_1) * (new_model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (1. / r1) * (sigma_t * (phi_1 / h - 1.)) * (new_model_s1 - model_s)
                )
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1}
        else:
            return x_t, model_s1

    def singlestep_dpm_solver_third_update(self, step, x, s, t, model_s_prev, r1=1./3., r2=2./3., model_s=None, model_s1=None, return_intermediate=False, solver_type='dpmsolver'):
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 1. / 3.
        if r2 is None:
            r2 = 2. / 3.
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)
        alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_12 = torch.expm1(-r2 * h)
            phi_1 = torch.expm1(-h)
            phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (
                    (sigma_s1 / sigma_s) * x
                    - (alpha_s1 * phi_11) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                (sigma_s2 / sigma_s) * x
                - (alpha_s2 * phi_12) * model_s
                + r2 / r1 * (alpha_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1. / r2) * (alpha_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (alpha_t * phi_2) * D1
                    - (alpha_t * phi_3) * D2
                )

        else:
            phi_11 = torch.expm1(r1 * h)
            phi_12 = torch.expm1(r2 * h)
            phi_1 = torch.expm1(h)
            phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s_prev != None:
                new_model_s = model_s + self.lambdas[step][0]*(model_s-model_s_prev)
            else:
                new_model_s = model_s + self.lambdas[step][0]*(model_s-x)
            x_s1 = (
                (torch.exp(log_alpha_s1 - log_alpha_s)) * x
                - (sigma_s1 * phi_11) * new_model_s
            )
            model_s1 = self.model_fn(x_s1, s1)

            new_model_s1 = model_s1 + self.lambdas[step][1]*(model_s1-model_s) 

            x_s2 = (
                (torch.exp(log_alpha_s2 - log_alpha_s)) * x
                - (sigma_s2 * phi_12) * model_s
                - r2 / r1 * (sigma_s2 * phi_22) * (new_model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)

            new_model_s2 = model_s2 + self.lambdas[step][2]*(model_s2-model_s1)

            if solver_type == 'dpmsolver':
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_s)) * x
                    - (sigma_t * phi_1) * model_s
                    - (1. / r2) * (sigma_t * phi_2) * (new_model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (new_model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_s)) * x
                    - (sigma_t * phi_1) * model_s
                    - (sigma_t * phi_2) * D1
                    - (sigma_t * phi_3) * D2
                )

        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1, 'model_s2': model_s2}
        else:
            return x_t, model_s2

    def singlestep_dpm_solver_update(self, step, x, s, t, order, model_s_prev, return_intermediate=False, solver_type='dpmsolver', r1=None, r2=None):
        """
        Singlestep DPM-Solver with the order `order` from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
            r1: A `float`. The hyperparameter of the second-order or third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(step, x, s, t, model_s_prev, return_intermediate=return_intermediate)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(step, x, s, t, model_s_prev, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1)
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(step, x, s, t, model_s_prev, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1, r2=r2)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def dpm_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5, solver_type='dpmsolver'):
        """
        The adaptive step size solver based on singlestep DPM-Solver.

        Args:
            x: A pytorch tensor. The initial value at time `t_T`.
            order: A `int`. The (higher) order of the solver. We only support order == 2 or 3.
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            h_init: A `float`. The initial step size (for logSNR).
            atol: A `float`. The absolute tolerance of the solver. For image data, the default setting is 0.0078, followed [1].
            rtol: A `float`. The relative tolerance of the solver. The default setting is 0.05.
            theta: A `float`. The safety hyperparameter for adapting the step size. The default setting is 0.9, followed [1].
            t_err: A `float`. The tolerance for the time. We solve the diffusion ODE until the absolute error between the 
                current time and `t_0` is less than `t_err`. The default setting is 1e-5.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        ns = self.noise_schedule
        s = t_T * torch.ones((1,)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_intermediate=True)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, solver_type=solver_type, **kwargs)
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, return_intermediate=True, solver_type=solver_type)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs)
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1. / order).float(), lambda_0 - lambda_s)
            nfe += order
        print('adaptive solver nfe', nfe)
        return x

    def add_noise(self, x, t, noise=None):
        """
        Compute the noised input xt = alpha_t * x + sigma_t * noise. 

        Args:
            x: A `torch.Tensor` with shape `(batch_size, *shape)`.
            t: A `torch.Tensor` with shape `(t_size,)`.
        Returns:
            xt with shape `(t_size, batch_size, *shape)`.
        """
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        if noise is None:
            noise = torch.randn((t.shape[0], *x.shape), device=x.device)
        x = x.reshape((-1, *x.shape))
        xt = expand_dims(alpha_t, x.dim()) * x + expand_dims(sigma_t, x.dim()) * noise
        if t.shape[0] == 1:
            return xt.squeeze(0)
        else:
            return xt

    def inverse(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        """
        Inverse the sample `x` from time `t_start` to `t_end` by DPM-Solver.
        For discrete-time DPMs, we use `t_start=1/N`, where `N` is the total time steps during training.
        """
        t_0 = 1. / self.noise_schedule.total_N if t_start is None else t_start
        t_T = self.noise_schedule.T if t_end is None else t_end
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        return self.sample(x, steps=steps, t_start=t_0, t_end=t_T, order=order, skip_type=skip_type,
            method=method, lower_order_final=lower_order_final, denoise_to_zero=denoise_to_zero, solver_type=solver_type,
            atol=atol, rtol=rtol, return_intermediate=return_intermediate)

    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='singlestep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in ['singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in ['singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []
        with torch.no_grad():
            if method in ['singlestep', 'singlestep_fixed']:
                if method == 'singlestep':
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device)
                elif method == 'singlestep_fixed':
                    K = steps // order
                    orders = [order,] * K
                    timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
                
                model_s_prev = None     
                for step, order in enumerate(orders):
                    s, t = timesteps_outer[step], timesteps_outer[step + 1]
                    timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device)
                    lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                    h = lambda_inner[-1] - lambda_inner[0]
                    r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                    r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
                    x, model_s_prev = self.singlestep_dpm_solver_update(step, x, s, t, order, model_s_prev, solver_type=solver_type, r1=r1, r2=r2)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
            else:
                raise ValueError("Got wrong method {}".format(method))
            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x





class Get_lambdas_DPM_Solver:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="dpmsolver",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.,
        dynamic_thresholding_ratio=0.995,
    ):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        self.algorithm_type = algorithm_type
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method. 
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0, t)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model. 
        """
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3,] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3,] * (K - 1) + [1]
            else:
                orders = [3,] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2,] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (K - 1) + [1]
        elif order == 1:
            K = 1
            orders = [1,] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == 'logSNR':
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[torch.cumsum(torch.tensor([0,] + orders), 0).to(device)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization. 
        """
        return self.data_prediction_fn(x, s)

    def get_lambdas(self, x, teacher_targets, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='singlestep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in ['singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in ['singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []
        lambdas = []
        with torch.no_grad():
            if method in ['singlestep', 'singlestep_fixed']:
                if method == 'singlestep':
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device)
                elif method == 'singlestep_fixed':
                    K = steps // order
                    orders = [order,] * K
                    timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)

                model_s_prev = None
                for step, (order, teacher_target)  in enumerate(zip(orders, teacher_targets)):
                    s, t = timesteps_outer[step], timesteps_outer[step + 1]
                    timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device)
                    lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                    h = lambda_inner[-1] - lambda_inner[0]                
                    r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                    r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
                    #get_lambdas
                    x, lambda_i, model_s_prev = self.singlestep_dpm_solver_update(step, x, teacher_target, s, t, model_s_prev, order, solver_type=solver_type, r1=r1, r2=r2)

                    lambdas.append(lambda_i)

                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
            else:
                raise ValueError("Got wrong method {}".format(method))
            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x, lambdas

    def singlestep_dpm_solver_update(self, step, x, teacher_target, s, t, model_s_prev, order, solver_type='dpmsolver', r1=None, r2=None):
        if order == 1:
            return self.dpm_solver_first_update(step, x, teacher_target, s, t, model_s_prev)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(step, x, teacher_target, s, t, model_s_prev, solver_type=solver_type, r1=r1)
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(step, x, teacher_target, s, t, model_s_prev, solver_type=solver_type, r1=r1, r2=r2)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def dpm_solver_first_update(self, step, x, teacher_target, s, t, model_s_prev, model_s=None):
        device = x.device
        
        ns = self.noise_schedule
        dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        phi_1 = torch.expm1(h)
        if model_s is None:
            model_s = self.model_fn(x, s)

        if model_s_prev == None:
            model_s_prev = x

        x_t = (
            torch.exp(log_alpha_t - log_alpha_s) * x
            - (sigma_t * phi_1) * model_s
        )

        #first lambdas
        x_target = teacher_target[0] - x_t.unsqueeze(0).to('cpu')
        e_prev = - (sigma_t * phi_1) * (model_s - model_s_prev).unsqueeze(0)
        first_lambdas, first_err = solve_ditillation_objective(e_prev.to('cpu'), x_target)
        #print(f'{step+1}th first step error: {first_err}', flush=True)
        new_x_t = x_t - first_lambdas.to(device)* (sigma_t * phi_1) * (model_s - model_s_prev)
        return new_x_t, [first_lambdas.to(device).item()], model_s



    def singlestep_dpm_solver_second_update(self, step, x, teacher_target, s, t, model_s_prev, r1=0.5, model_s=None, solver_type='dpmsolver'):
        device = x.device
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)


        phi_11 = torch.expm1(r1 * h)
        phi_1 = torch.expm1(h)

        if model_s is None:
            model_s = self.model_fn(x, s)

        if model_s_prev == None:
            model_s_prev = x

        x_s1 = (
            torch.exp(log_alpha_s1 - log_alpha_s) * x
            - (sigma_s1 * phi_11) * model_s
        )

        #first lambdas
        x_target = teacher_target[0] - x_s1.unsqueeze(0).to('cpu')
        e_prev = - (sigma_s1 * phi_11) *(model_s - model_s_prev).unsqueeze(0)
        first_lambdas, first_err = solve_ditillation_objective(e_prev.to('cpu'), x_target)
        #print(f'{step+1}th first step error: {first_err}', flush=True)
        new_x_s1 = x_s1 - first_lambdas.to(device)*(sigma_s1 * phi_11) *(model_s - model_s_prev)
        model_s1 = self.model_fn(new_x_s1, s1)

        #second lambdas
        x_t = (
            torch.exp(log_alpha_t - log_alpha_s) * x
            - (sigma_t * phi_1) * model_s
            - (0.5 / r1) * (sigma_t * phi_1) * (model_s1 - model_s)
        )
        x_target = teacher_target[1] - x_t.unsqueeze(0).to('cpu')
        e_prev = - (0.5 / r1) * (sigma_t * phi_1)  *(model_s1 - model_s).unsqueeze(0)
        second_lambdas, second_err = solve_ditillation_objective(e_prev.to('cpu'), x_target)
        #print(f'{step+1}th second step error: {second_err}', flush=True)
        new_x_t = x_t - second_lambdas.to(device)*(0.5 / r1) * (sigma_t * phi_1)  *(model_s1 - model_s)
        return new_x_t, [first_lambdas.to(device).item(), second_lambdas.to(device).item()], model_s1       


    def singlestep_dpm_solver_third_update(self, step, x, teacher_target, s, t, model_s_prev, r1=1./3., r2=2./3., model_s=None, model_s1=None, solver_type='dpmsolver'):
        device = x.device
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 1. / 3.
        if r2 is None:
            r2 = 2. / 3.
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)
        alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)


        phi_11 = torch.expm1(r1 * h)
        phi_12 = torch.expm1(r2 * h)
        phi_1 = torch.expm1(h)
        phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.
        phi_2 = phi_1 / h - 1.
        phi_3 = phi_2 / h - 0.5

        if model_s is None:
            model_s = self.model_fn(x, s)

        if model_s_prev == None:
            model_s_prev = x

        x_s1 = (
            (torch.exp(log_alpha_s1 - log_alpha_s)) * x
            - (sigma_s1 * phi_11) * model_s
        )
        #first lambdas
        x_target = teacher_target[0] - x_s1.unsqueeze(0).to('cpu')
        e_prev = - (sigma_s1 * phi_11) *(model_s - model_s_prev).unsqueeze(0)
        first_lambdas, first_err = solve_ditillation_objective(e_prev.to('cpu'), x_target)
        #print(f'{step+1}th first step error: {first_err}', flush=True)
        new_x_s1 = x_s1 - first_lambdas.to(device)*(sigma_s1 * phi_11) *(model_s - model_s_prev)
        model_s1 = self.model_fn(new_x_s1, s1)

        #second lambdas
        x_s2 = (
            (torch.exp(log_alpha_s2 - log_alpha_s)) * x
            - (sigma_s2 * phi_12) * model_s
            - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_s)
        )
        x_target = teacher_target[1] - x_s2.unsqueeze(0).to('cpu')
        e_prev = - r2 / r1 * (sigma_s2 * phi_22) *(model_s1 - model_s).unsqueeze(0)
        second_lambdas, second_err = solve_ditillation_objective(e_prev.to('cpu'), x_target)
        #print(f'{step+1}th second step error: {second_err}', flush=True)
        new_x_s2 = x_s2 - second_lambdas.to(device)* (r2 / r1) * (sigma_s2 * phi_22) *(model_s1 - model_s)
        model_s2 = self.model_fn(new_x_s2, s2)

        #third lambdas
        x_t = (
            (torch.exp(log_alpha_t - log_alpha_s)) * x
            - (sigma_t * phi_1) * model_s
            - (1. / r2) * (sigma_t * phi_2) * (model_s2 - model_s)
        )
        x_target = teacher_target[2] - x_t.unsqueeze(0).to('cpu')
        e_prev = - (1. / r2) * (sigma_t * phi_2) *(model_s2 - model_s1).unsqueeze(0)
        third_lambdas, third_err = solve_ditillation_objective(e_prev.to('cpu'), x_target)
        #print(f'{step+1}th third step error: {third_err}', flush=True)
        new_x_t = x_t - third_lambdas.to(device) * (1. / r2) * (sigma_t * phi_2) *(model_s2 - model_s1)

        return new_x_t, [first_lambdas.to(device).item(), second_lambdas.to(device).item(), third_lambdas.to(device).item()], model_s2        

        

#############################################################
# other utility functions
#############################################################

def solve_ditillation_objective(e_prev, x_target):
    e_prev = e_prev.reshape(e_prev.shape[0], -1)
    x_target = x_target.flatten()
    # solve the least squares problem to obtain lambdas
    lambdas = torch.linalg.lstsq(e_prev.T.to(torch.float64), x_target.to(torch.float64)).solution
    # error
    x_pred = torch.matmul(lambdas.unsqueeze(0), e_prev.to(torch.float64)).squeeze(0)
    err = torch.nn.functional.mse_loss(x_pred, x_target).tolist()
    return lambdas.float(), err

def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,)*(dims - 1)]
import numpy as np
import torch

def solve_ditillation_objective(denoised_prev, x_target):
    denoised_prev = denoised_prev.unsqueeze(0)
    denoised_prev = denoised_prev.reshape(denoised_prev.shape[0], -1)
    x_target = x_target.flatten()
    # solve the least squares problem to obtain lambdas
    lambdas = torch.linalg.lstsq(denoised_prev.T.to(torch.float64), x_target.to(torch.float64)).solution
    # error
    x_pred = torch.matmul(lambdas.unsqueeze(0), denoised_prev.to(torch.float64)).squeeze(0)
    err = torch.nn.functional.mse_loss(x_pred, x_target).tolist()
    return lambdas, err

#----------------------------------------------------------------------------
# Distilled DDIM sampler.

def d_ddim_sampler(
    lambdas, net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7, scale=10,
    **sampler_kwargs
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    if lambdas == None: # At first batch, we don't have lambdas yet.
        # Teacher sampling DDIM with "teacher_steps" steps
        teacher_steps = scale*num_steps
        teacher_step_indices = torch.arange(teacher_steps, dtype=torch.float64, device=latents.device)
        teacher_t_steps = (sigma_max ** (1 / rho) + teacher_step_indices / (teacher_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        teacher_t_steps = torch.cat([net.round_sigma(teacher_t_steps), torch.zeros_like(teacher_t_steps[:1])]) # t_N = 0

        x_next = latents.to(torch.float64) * teacher_t_steps[0]
        xs = [x_next.to('cpu')]
        for i, (t_cur, t_next) in enumerate(zip(teacher_t_steps[:-1], teacher_t_steps[1:])): # 0, ..., N-1
            x_cur = x_next
            dt = t_next - t_cur
            denoised = net(x_cur, t_cur, class_labels).to(torch.float64)

            x_next = x_cur + dt*(x_cur-denoised)/t_cur

            xs.append(x_next.to('cpu'))

        # Set teacher targets
        teacher_targets = []
        for i in range(num_steps):
            teacher_targets.append(xs[(i+1)*scale])
        teacher_targets = torch.stack(teacher_targets, dim=0)

        # Get lambdas
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        lambdas = torch.zeros(num_steps, dtype=torch.float64)
        denoised_list = []
        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next
            dt = t_next - t_cur
            denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
            d_dir = (x_cur-denoised)/t_cur

            x_next = x_cur + dt*d_dir

            if len(denoised_list) > 0:
                x_target = teacher_targets[i:i+1].to(latents.device) - x_next.unsqueeze(0)
                denoised_prev = denoised - denoised_list[-1].to(latents.device)             

                # Optimize lambda
                lambdas[i], err = solve_ditillation_objective(denoised_prev, x_target) 
                #print(f'{i+1}th step error: {err}')

                # Re-estimate next sample with optimized lambda
                x_next += lambdas[i]*denoised_prev

            denoised_list.append(denoised.to('cpu'))

    else: # From the second batch we reuse lambdas
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        denoised_list = []
        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next
            dt = t_next - t_cur
            denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
            d_dir = (x_cur-denoised)/t_cur

            x_next = x_cur + dt*d_dir

            if len(denoised_list) > 0:
                denoised_prev = denoised - denoised_list[-1].to(latents.device)            
                x_next += lambdas[i]*denoised_prev

            denoised_list.append(denoised.to('cpu'))

    return lambdas, x_next



#----------------------------------------------------------------------------
# Distilled EDM sampler.

def d_edm_sampler(
    lambdas, net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, scale=10,
    **sampler_kwargs
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    if lambdas == None: # At first batch, we don't have lambdas yet.
        # Teacher sampling DDIM with "teacher_steps" steps
        NFE = 2*num_steps -1
        teacher_steps = scale*NFE
        teacher_step_indices = torch.arange(teacher_steps, dtype=torch.float64, device=latents.device)
        teacher_t_steps = (sigma_max ** (1 / rho) + teacher_step_indices / (teacher_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        teacher_t_steps = torch.cat([net.round_sigma(teacher_t_steps), torch.zeros_like(teacher_t_steps[:1])]) # t_N = 0

        x_next = latents.to(torch.float64) * teacher_t_steps[0]
        xs = [x_next.to('cpu')]
        for i, (t_cur, t_next) in enumerate(zip(teacher_t_steps[:-1], teacher_t_steps[1:])): # 0, ..., N-1
            x_cur = x_next
            dt = t_next - t_cur
            denoised = net(x_cur, t_cur, class_labels).to(torch.float64)

            x_next = x_cur + dt*(x_cur-denoised)/t_cur

            xs.append(x_next.to('cpu'))

        # num_steps = 5  NFE 9, 
        # teacher_steps = 90 , len(xs) = 91 (including xT and x0) 
        # teacher_targets = [(idx10, idx20), (idx30, idx40), (idx50, idx60), (idx70, idx80), (idx90)]
        # Set teacher targets
        teacher_targets = []
        for i in range(num_steps):
            if i >= num_steps - 1:
                teacher_targets.append((xs[(2*i+1)*scale].unsqueeze(0)))
            else:
                teacher_targets.append((xs[(2*i+1)*scale].unsqueeze(0), xs[(2*i+2)*scale].unsqueeze(0)))

        #get lambdas
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        lambdas = torch.zeros(NFE, dtype=torch.float64)
        denoised_list = []
        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if len(denoised_list) > 0:
                x_target = teacher_targets[i][0].to(latents.device) - x_next.unsqueeze(0) 

                denoised_prev = denoised - denoised_list[-1].to(latents.device)

                # Optimize lambda
                lambdas[2*i], err = solve_ditillation_objective(denoised_prev, x_target)    
                #print(f'{2*i}th step error: {err}', flush=True)

                # Re-estimate next sample with optimized lambda
                x_next += lambdas[2*i]*denoised_prev

            denoised_list.append(denoised.to('cpu'))

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)   

                x_target = teacher_targets[i][1].to(latents.device) - x_next.unsqueeze(0) 

                denoised_prev = denoised - denoised_list[-1].to(latents.device)

                # Optimize lambda
                lambdas[2*i+1], err = solve_ditillation_objective(denoised_prev, x_target)    
                #print(f'{2*i+1}th step error: {err}', flush=True)

                # Re-estimate next sample with optimized lambda
                x_next += lambdas[2*i+1]*denoised_prev.squeeze()

                denoised_list.append(denoised.to('cpu'))

        return lambdas, x_next

    else: # From the second batch we reuse lambdas
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        denoised_list = []
        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if len(denoised_list)> 0:
                denoised_prev = denoised - denoised_list[-1].to(latents.device)
                x_next += lambdas[2*i]*denoised_prev
            denoised_list.append(denoised.to('cpu'))

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)   

                denoised_prev = denoised - denoised_list[-1].to(latents.device)
                x_next += lambdas[2*i+1]*denoised_prev
                denoised_list.append(denoised.to('cpu'))

        return lambdas, x_next
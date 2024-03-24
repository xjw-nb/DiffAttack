# ---------------------------------------------------------------
# https://github.com/w86763777/pytorch-ddpm/blob/master/diffusion.py
# ---------------------------------------------------------------

import os
import random

import numpy as np

import torch
import torchvision.utils as tvu

from ddpm.unet_cifar_ddpm import UNet

import torch.nn as nn
import torch.nn.functional as F


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))




class Diffusion_cifar(torch.nn.Module):
    def __init__(self, args, device=None):
        super().__init__()

        self.args = args

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        print("Loading model")
        model = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],num_res_blocks=2, dropout=0.1)
        ckpt = torch.load('./pretrained/ckpt.pt')
        model.load_state_dict(ckpt['net_model'])
        self.model = model.to(device)


        # args of ddpm
        mean_type = 'epsilon'
        var_type = 'fixedlarge'
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        self.T = 1000
        self.img_size = 32
        self.mean_type = mean_type
        self.var_type = var_type
        beta_1 = 0.0001
        beta_T = 0.02
        self.sample_step=1

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, self.T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:self.T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model(x_t, t)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        return torch.clip(x_0, -1, 1)

    def image_editing_sample(self, img=None, bs_id=0, tag=None):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]


        if tag is None:
            tag = 'rnd' + str(random.randint(0, 10000))
        # out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)

        assert img.ndim == 4, img.ndim
        x0 = img

        # if bs_id < 2:
        #     os.makedirs(out_dir, exist_ok=True)
        #     tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'original_input.png'))

        xs = []
        for it in range(self.sample_step):
            e = torch.randn_like(x0)
            total_noise_levels = self.args.t
            a = (1 - self.betas).cumprod(dim=0).to(x0.device)
            x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

            ori_x = []
            for t_tmp in reversed(range(0, total_noise_levels - 1)):
                ori_x.append(x0 * a[t_tmp].sqrt() + e * (1.0 - a[t_tmp]).sqrt())
            ori_x.append(x0)

            # if bs_id < 2:
            #     tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'init_{it}.png'))

            mid_x = []
            for i in reversed(range(total_noise_levels)):
                t = torch.tensor([i] * batch_size, device=img.device)

                x, log_var = self.p_mean_variance(x_t=x, t=t)
                # no noise when t == 0
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0
                x = x + torch.exp(0.5 * log_var) * noise

                mid_x.append(x)



            x0 = x
            x0 = torch.clip(x0, -1, 1)

            # if bs_id < 2:
            #     torch.save(x0, os.path.join(out_dir, f'samples_{it}.pth'))
            #     tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'samples_{it}.png'))

            xs.append(x0)

            return torch.cat(xs, dim=0), mid_x, ori_x





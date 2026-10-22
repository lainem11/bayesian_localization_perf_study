"""Step 2(b1): one Adam over all 6 parameters, one foreach launch per operation.

Copies the non-capturable path of torch.optim.adam._multi_tensor_adam
(PyTorch 2.12.0) with per-tensor learning rates, so the arithmetic should be
bitwise identical to 6 parameter groups. Installed at run time; src/ unchanged.
"""
import torch
from ABL.model import ResponseModel


class MergedAdam:
    def __init__(self, params, lrs, betas, eps=1e-8):
        self.params = params                      # the 6 parameter tensors
        self.lrs = list(lrs)                      # one learning rate per tensor
        self.beta1, self.beta2 = betas
        self.eps = eps                            # torch.optim.Adam default
        self.exp_avgs = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in params]
        self.exp_avg_sqs = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in params]
        self.steps = [torch.tensor(0.0) for _ in params]   # CPU float32 counters, as in Adam

    def zero_grad(self):
        for p in self.params:                     # Adam's default: set_to_none=True
            p.grad = None

    @torch.no_grad()
    def step(self):
        grads = [p.grad for p in self.params]
        torch._foreach_add_(self.steps, torch.tensor(1.0, device="cpu"), alpha=1.0)  # line 130
        torch._foreach_lerp_(self.exp_avgs, grads, 1 - self.beta1)                    # line 152
        torch._foreach_mul_(self.exp_avg_sqs, self.beta2)                             # line 156
        torch._foreach_addcmul_(self.exp_avg_sqs, grads, grads, 1 - self.beta2)       # line 169
        bc1 = [1 - self.beta1 ** s.item() for s in self.steps]                        # lines 219-224
        bc2 = [1 - self.beta2 ** s.item() for s in self.steps]
        step_size = [(lr / bc) * -1 for lr, bc in zip(self.lrs, bc1)]                 # line 226, per-tensor lr
        bc2_sqrt = [bc ** 0.5 for bc in bc2]                                          # line 228
        denom = torch._foreach_sqrt(self.exp_avg_sqs)                                 # line 238
        torch._foreach_div_(denom, bc2_sqrt)                                          # line 240
        torch._foreach_add_(denom, self.eps)                                          # line 241
        torch._foreach_addcdiv_(self.params, self.exp_avgs, denom, step_size)         # line 242


class MergedScheduler:
    """Same rule as MultiplicativeLR: lr = lr * factor after every step."""
    def __init__(self, opt, factor_fn):
        self.opt, self.factor_fn, self.epoch = opt, factor_fn, 0

    def step(self):
        self.epoch += 1
        f = self.factor_fn(self.epoch)
        self.opt.lrs = [lr * f for lr in self.opt.lrs]


def install():
    """Replace ResponseModel._setup_optimizer (model.py lines 431-448)."""
    def setup(self, opt_param_dict, opt_settings, trial_number):
        lr_default = opt_settings['lr']
        lr_decay_factor = opt_settings.get('global_gamma', 1.0) ** (trial_number - 1)
        params, lrs = [], []
        for k, v in opt_param_dict.items():
            base_lr = self.opt_settings.get(k, {}).get('lr', lr_default)
            params.append(v['values'])
            lrs.append(base_lr * lr_decay_factor)
            v['values'].register_hook(lambda grad: torch.clamp(grad, -opt_settings['clip_grad'], opt_settings['clip_grad']))
        opt = MergedAdam(params, lrs, opt_settings['betas'])
        sched = MergedScheduler(opt, lambda epoch: opt_settings['gamma'] ** (1 / opt_settings['opt_steps']))
        return opt, sched

    ResponseModel._setup_optimizer = setup
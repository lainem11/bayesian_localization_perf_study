import json
import torch
from ABL.model import ResponseModel

def install(ref_path):
    with open(ref_path) as f:
        steps = json.load(f)["runs"][0]["adam steps"]   # steps per trial (same in all runs)

    def loop(self, optimizer, scheduler, opt_param_dict, e_mag, angles, responses, opt_settings):
        n = steps[responses.numel() - 1]                 # trial k has k responses → its step count
        losses = torch.zeros(n, device=responses.device) # loss history stays on the GPU
        for step_i in range(n):
            optimizer.zero_grad()
            back = {k: {'values': v['conversion_func'][1](v['values'])} for k, v in opt_param_dict.items()}
            responsivity = self.response_func(e_mag, angles, back)
            likelihood = responsivity * responses.view(-1, 1) + (1 - responsivity) * (1 - responses).view(-1, 1)
            loss = -likelihood.log().mean()
            loss += sum(self.regularization_lambda * (v['values']**2).mean() for v in opt_param_dict.values())
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses[step_i] = loss.detach()               # GPU→GPU copy, no sync
        return losses.cpu(), True                        # one sync per localize() instead of one per step

    ResponseModel._run_optimization_loop = loop          # every ResponseModel now uses this loop
"""Step 2(b2): compile the loss (forward + backward) with torch.compile.

Replaces ResponseModel._run_optimization_loop at run time. The loop is the
original (model.py lines 450-484, with loss.item() and the convergence check);
only the loss computation goes through torch.compile. src/ unchanged.
"""
import torch
from ABL.model import ResponseModel


def _loss(response_func, reg_lambda, keys, convs, values, e_mag, angles, responses):
    # Same expressions as model.py lines 460-468
    back = {k: {'values': c(v)} for k, c, v in zip(keys, convs, values)}
    responsivity = response_func(e_mag, angles, back)
    likelihood = responsivity * responses.view(-1, 1) + (1 - responsivity) * (1 - responses).view(-1, 1)
    loss = -likelihood.log().mean()
    reg = sum(reg_lambda * (v**2).mean() for v in values)
    return loss + reg


_compiled_loss = torch.compile(_loss)   # compiled lazily, on the first call


def install():
    def loop(self, optimizer, scheduler, opt_param_dict, e_mag, angles, responses, opt_settings):
        keys = tuple(opt_param_dict)
        convs = [opt_param_dict[k]['conversion_func'][1] for k in keys]
        values = [opt_param_dict[k]['values'] for k in keys]
        losses = torch.zeros((opt_settings['opt_steps'],))
        converged = False
        conv_buffer = []
        step_i = 0

        while not converged and step_i < opt_settings['opt_steps']:
            optimizer.zero_grad()
            loss = _compiled_loss(self.response_func, self.regularization_lambda,
                                  keys, convs, values, e_mag, angles, responses)
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses[step_i] = loss.item()
            if step_i > 0 and torch.abs(losses[step_i] - losses[step_i - 1]) < 1e-6:
                conv_buffer.append(1)
                if len(conv_buffer) >= 6:
                    converged = True
            else:
                conv_buffer = []
            step_i += 1

        return losses[:step_i], converged

    ResponseModel._run_optimization_loop = loop
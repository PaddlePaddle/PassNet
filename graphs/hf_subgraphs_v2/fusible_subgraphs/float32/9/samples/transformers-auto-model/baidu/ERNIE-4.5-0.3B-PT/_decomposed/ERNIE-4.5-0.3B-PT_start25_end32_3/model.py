import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.cat((in_0, in_0), dim=-1)
        tmp_1 = tmp_0.cos()
        tmp_2 = tmp_1 * 1.0
        tmp_1 = None
        tmp_3 = tmp_0.sin()
        tmp_0 = None
        tmp_4 = tmp_3 * 1.0
        tmp_3 = None
        tmp_5 = torch.set_grad_enabled(True)
        tmp_5 = None
        tmp_6 = torch._C._log_api_usage_once('python.nn_module')
        tmp_6 = None
        return (tmp_2, tmp_4)
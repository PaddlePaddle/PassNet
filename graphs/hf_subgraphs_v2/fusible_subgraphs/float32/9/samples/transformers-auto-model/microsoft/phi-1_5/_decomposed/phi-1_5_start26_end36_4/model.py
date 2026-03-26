import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.cat((in_2, in_2), dim=-1)
        tmp_3 = tmp_2.cos()
        tmp_4 = tmp_3 * 1.0
        tmp_3 = None
        tmp_5 = tmp_2.sin()
        tmp_2 = None
        tmp_6 = tmp_5 * 1.0
        tmp_5 = None
        tmp_7 = tmp_4.to(dtype=torch.float16)
        tmp_4 = None
        tmp_8 = tmp_6.to(dtype=torch.float16)
        tmp_6 = None
        tmp_9 = torch.set_grad_enabled(True)
        tmp_9 = None
        tmp_10 = torch._C._log_api_usage_once('python.nn_module')
        tmp_10 = None
        tmp_11 = torch.nn.functional.layer_norm(in_3, (2048,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        return (tmp_7, tmp_11, tmp_8)
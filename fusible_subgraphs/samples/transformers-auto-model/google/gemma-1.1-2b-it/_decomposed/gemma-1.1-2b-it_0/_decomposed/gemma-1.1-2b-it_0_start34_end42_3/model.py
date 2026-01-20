import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0 * in_1
        tmp_1 = torch._C._log_api_usage_once('python.nn_module')
        tmp_1 = None
        tmp_2 = tmp_0.float()
        tmp_3 = tmp_2.pow(2)
        tmp_4 = tmp_3.mean(-1, keepdim=True)
        tmp_3 = None
        tmp_5 = tmp_4 + 1e-06
        tmp_4 = None
        tmp_6 = torch.rsqrt(tmp_5)
        tmp_5 = None
        tmp_7 = tmp_2 * tmp_6
        tmp_2 = tmp_6 = None
        return (tmp_0, tmp_7)
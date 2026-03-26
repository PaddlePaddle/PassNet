import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = tmp_0 * in_2
        tmp_0 = None
        tmp_3 = torch._C._log_api_usage_once('python.nn_module')
        tmp_3 = None
        tmp_4 = tmp_2.float()
        tmp_5 = tmp_4.pow(2)
        tmp_6 = tmp_5.mean(-1, keepdim=True)
        tmp_5 = None
        tmp_7 = tmp_6 + 1e-06
        tmp_6 = None
        tmp_8 = torch.rsqrt(tmp_7)
        tmp_7 = None
        tmp_9 = tmp_4 * tmp_8
        tmp_4 = tmp_8 = None
        tmp_10 = tmp_1.float()
        tmp_1 = None
        tmp_11 = 1.0 + tmp_10
        tmp_10 = None
        tmp_12 = tmp_9 * tmp_11
        tmp_9 = tmp_11 = None
        tmp_13 = tmp_12.type_as(tmp_2)
        tmp_12 = None
        return (tmp_2, tmp_13)
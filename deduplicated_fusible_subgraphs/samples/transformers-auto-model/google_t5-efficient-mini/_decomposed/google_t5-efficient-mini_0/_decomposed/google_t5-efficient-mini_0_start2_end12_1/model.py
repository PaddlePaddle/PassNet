import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, w_0):
        tmp_0 = in_1 + in_2
        tmp_1 = in_0[-1]
        tmp_2 = tmp_1 + 1
        tmp_1 = tmp_2 = None
        tmp_3 = tmp_0.to(torch.float32)
        tmp_4 = tmp_3.pow(2)
        tmp_3 = None
        tmp_5 = tmp_4.mean(-1, keepdim=True)
        tmp_4 = None
        tmp_6 = tmp_5 + 1e-06
        tmp_5 = None
        tmp_7 = torch.rsqrt(tmp_6)
        tmp_6 = None
        tmp_8 = tmp_0 * tmp_7
        tmp_7 = None
        tmp_9 = w_0 * tmp_8
        tmp_8 = None
        return (tmp_0, tmp_9)
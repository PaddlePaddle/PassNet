import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_3 + in_2
        tmp_2 = in_1[-1]
        tmp_3 = tmp_2 + 1
        tmp_2 = tmp_3 = None
        tmp_4 = tmp_1.to(torch.float32)
        tmp_5 = tmp_4.pow(2)
        tmp_4 = None
        tmp_6 = tmp_5.mean(-1, keepdim=True)
        tmp_5 = None
        tmp_7 = tmp_6 + 1e-06
        tmp_6 = None
        tmp_8 = torch.rsqrt(tmp_7)
        tmp_7 = None
        tmp_9 = tmp_1 * tmp_8
        tmp_8 = None
        tmp_10 = tmp_0 * tmp_9
        tmp_0 = tmp_9 = None
        return (tmp_1, tmp_10)
import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_2 + in_1
        tmp_2 = tmp_1.float()
        tmp_3 = tmp_2.pow(2)
        tmp_4 = tmp_3.mean(-1, keepdim=True)
        tmp_3 = None
        tmp_5 = tmp_4 + 1e-06
        tmp_4 = None
        tmp_6 = torch.rsqrt(tmp_5)
        tmp_5 = None
        tmp_7 = tmp_2 * tmp_6
        tmp_2 = tmp_6 = None
        tmp_8 = tmp_0.float()
        tmp_0 = None
        tmp_9 = 1.0 + tmp_8
        tmp_8 = None
        tmp_10 = tmp_7 * tmp_9
        tmp_7 = tmp_9 = None
        tmp_11 = tmp_10.type_as(tmp_1)
        tmp_10 = tmp_1 = None
        return (tmp_11,)
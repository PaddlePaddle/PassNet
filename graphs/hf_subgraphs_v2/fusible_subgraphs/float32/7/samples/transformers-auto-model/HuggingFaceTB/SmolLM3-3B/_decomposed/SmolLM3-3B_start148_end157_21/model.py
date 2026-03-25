import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_2 + in_1
        tmp_2 = tmp_1.to(torch.float32)
        tmp_3 = tmp_2.pow(2)
        tmp_4 = tmp_3.mean(-1, keepdim=True)
        tmp_3 = None
        tmp_5 = tmp_4 + 1e-06
        tmp_4 = None
        tmp_6 = torch.rsqrt(tmp_5)
        tmp_5 = None
        tmp_7 = tmp_2 * tmp_6
        tmp_2 = tmp_6 = None
        tmp_8 = tmp_7.to(torch.bfloat16)
        tmp_7 = None
        tmp_9 = tmp_0 * tmp_8
        tmp_0 = tmp_8 = None
        return (tmp_1, tmp_9)
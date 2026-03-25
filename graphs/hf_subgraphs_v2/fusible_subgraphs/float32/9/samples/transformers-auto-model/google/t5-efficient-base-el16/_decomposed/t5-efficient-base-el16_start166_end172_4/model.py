import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_1 + in_0
        tmp_1 = tmp_0.to(torch.float32)
        tmp_2 = tmp_1.pow(2)
        tmp_1 = None
        tmp_3 = tmp_2.mean(-1, keepdim=True)
        tmp_2 = None
        tmp_4 = tmp_3 + 1e-06
        tmp_3 = None
        tmp_5 = torch.rsqrt(tmp_4)
        tmp_4 = None
        return (tmp_0, tmp_5)
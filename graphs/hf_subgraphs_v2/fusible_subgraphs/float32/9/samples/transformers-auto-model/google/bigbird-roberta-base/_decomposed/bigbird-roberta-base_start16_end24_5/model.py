import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0 - in_1
        tmp_1 = tmp_0.split(1, dim=-1)
        tmp_0 = None
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_1 = None
        tmp_4 = tmp_2.squeeze(-1)
        tmp_2 = None
        tmp_5 = tmp_4.contiguous()
        tmp_4 = None
        tmp_6 = tmp_3.squeeze(-1)
        tmp_3 = None
        tmp_7 = tmp_6.contiguous()
        tmp_6 = None
        return (tmp_5, tmp_7)
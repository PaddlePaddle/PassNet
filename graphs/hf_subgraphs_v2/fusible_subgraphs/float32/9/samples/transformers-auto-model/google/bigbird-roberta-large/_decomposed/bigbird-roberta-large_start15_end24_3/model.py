import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = tmp_0 * 1000000.0
        tmp_0 = None
        tmp_2 = in_1 - tmp_1
        tmp_1 = None
        tmp_3 = tmp_2.split(1, dim=-1)
        tmp_2 = None
        tmp_4 = tmp_3[0]
        tmp_5 = tmp_3[1]
        tmp_3 = None
        tmp_6 = tmp_4.squeeze(-1)
        tmp_4 = None
        tmp_7 = tmp_6.contiguous()
        tmp_6 = None
        tmp_8 = tmp_5.squeeze(-1)
        tmp_5 = None
        tmp_9 = tmp_8.contiguous()
        tmp_8 = None
        return (tmp_7, tmp_9)
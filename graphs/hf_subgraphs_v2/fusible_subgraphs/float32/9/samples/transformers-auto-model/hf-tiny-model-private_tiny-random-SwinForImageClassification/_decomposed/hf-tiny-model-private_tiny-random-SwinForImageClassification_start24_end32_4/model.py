import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2 / 2.8284271247461903
        tmp_3 = tmp_0.view(-1)
        tmp_0 = None
        tmp_4 = tmp_1[tmp_3]
        tmp_1 = tmp_3 = None
        tmp_5 = tmp_4.view(4, 4, -1)
        tmp_4 = None
        tmp_6 = tmp_5.permute(2, 0, 1)
        tmp_5 = None
        tmp_7 = tmp_6.contiguous()
        tmp_6 = None
        tmp_8 = tmp_7.unsqueeze(0)
        tmp_7 = None
        tmp_9 = tmp_2 + tmp_8
        tmp_2 = tmp_8 = None
        return (tmp_9,)
import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, w_0, w_1):
        tmp_0 = torch.nn.functional.linear(in_0, w_1, w_0)
        tmp_1 = tmp_0.view((1, 45, 4, 8))
        tmp_0 = None
        tmp_2 = tmp_1.permute(0, 2, 1, 3)
        tmp_1 = None
        tmp_3 = in_1.view((1, 45, 4, 8))
        tmp_4 = tmp_3.permute(0, 2, 1, 3)
        tmp_3 = None
        tmp_5 = in_2.transpose(-1, -2)
        return (tmp_2, tmp_4, tmp_5)
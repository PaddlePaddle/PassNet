import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.nn.functional.linear(tmp_0, tmp_2, tmp_1)
        tmp_0 = tmp_2 = tmp_1 = None
        tmp_4 = in_3.view(1, 10, -1, 64)
        tmp_5 = tmp_4.transpose(1, 2)
        tmp_4 = None
        tmp_6 = tmp_3.view(1, 10, -1, 64)
        tmp_3 = None
        tmp_7 = tmp_6.transpose(1, 2)
        tmp_6 = None
        return (tmp_5, tmp_7)
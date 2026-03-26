import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(tmp_0, tmp_1, None)
        tmp_0 = tmp_1 = None
        tmp_3 = in_2.view(1, -1, 4, 64)
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = tmp_2.view(1, -1, 4, 64)
        tmp_2 = None
        tmp_6 = tmp_5.transpose(1, 2)
        tmp_5 = None
        tmp_7 = tmp_4.transpose(3, 2)
        return (tmp_4, tmp_7, tmp_6)
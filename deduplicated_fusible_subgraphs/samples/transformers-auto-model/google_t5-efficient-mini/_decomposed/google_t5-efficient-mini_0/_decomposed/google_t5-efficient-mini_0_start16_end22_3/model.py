import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0):
        tmp_0 = torch.nn.functional.linear(in_0, w_0, None)
        tmp_1 = in_1.view(1, -1, 8, 64)
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        tmp_3 = tmp_0.view(1, -1, 8, 64)
        tmp_0 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = tmp_2.transpose(3, 2)
        return (tmp_2, tmp_4, tmp_5)
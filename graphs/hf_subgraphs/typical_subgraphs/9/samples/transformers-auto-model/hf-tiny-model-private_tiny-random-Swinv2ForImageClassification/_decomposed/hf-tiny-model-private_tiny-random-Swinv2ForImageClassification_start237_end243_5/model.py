import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = torch.nn.functional.linear(in_0, tmp_2, tmp_1)
        tmp_2 = tmp_1 = None
        tmp_4 = tmp_3.view(4, -1, 4, 16)
        tmp_3 = None
        tmp_5 = tmp_4.transpose(1, 2)
        tmp_4 = None
        tmp_6 = torch.nn.functional.linear(in_0, tmp_0, None)
        tmp_0 = None
        tmp_7 = tmp_6.view(4, -1, 4, 16)
        tmp_6 = None
        tmp_8 = tmp_7.transpose(1, 2)
        tmp_7 = None
        return (tmp_8, tmp_5)
import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.linear(in_0, tmp_1, None)
        tmp_1 = None
        tmp_3 = tmp_2.view((1, 2, -1, 128))
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = torch.nn.functional.linear(in_0, tmp_0, None)
        tmp_0 = None
        tmp_6 = tmp_5.view((1, 2, -1, 128))
        tmp_5 = None
        tmp_7 = tmp_6.transpose(1, 2)
        tmp_6 = None
        return (tmp_7, tmp_4)
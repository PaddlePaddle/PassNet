import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = in_3.view(1, -1, 4, 4)
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = tmp_2.view(1, -1, 4, 4)
        tmp_2 = None
        tmp_6 = tmp_5.transpose(1, 2)
        tmp_5 = None
        tmp_7 = in_4.view(1, 22, 4, 4)
        tmp_8 = tmp_7.transpose(1, 2)
        tmp_7 = None
        tmp_9 = tmp_8.reshape(4, -1, 4)
        tmp_8 = None
        tmp_10 = tmp_4.reshape(4, -1, 4)
        tmp_11 = tmp_6.reshape(4, -1, 4)
        tmp_12 = tmp_10.transpose(1, 2)
        tmp_10 = None
        return (tmp_4, tmp_9, tmp_12, tmp_6, tmp_11)
import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.nn.functional.linear(tmp_0, tmp_2, tmp_1)
        tmp_0 = tmp_2 = tmp_1 = None
        tmp_4 = in_3.view(1, -1, 4, 4)
        tmp_5 = tmp_4.transpose(1, 2)
        tmp_4 = None
        tmp_6 = tmp_3.view(1, -1, 4, 4)
        tmp_3 = None
        tmp_7 = tmp_6.transpose(1, 2)
        tmp_6 = None
        tmp_8 = in_4.view(1, 22, 4, 4)
        tmp_9 = tmp_8.transpose(1, 2)
        tmp_8 = None
        tmp_10 = tmp_9.reshape(4, -1, 4)
        tmp_9 = None
        tmp_11 = tmp_5.reshape(4, -1, 4)
        tmp_12 = tmp_7.reshape(4, -1, 4)
        tmp_13 = tmp_11.transpose(1, 2)
        tmp_11 = None
        return (tmp_5, tmp_10, tmp_13, tmp_7, tmp_12)
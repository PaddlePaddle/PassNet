import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, w_0, w_1):
        tmp_0 = torch.nn.functional.linear(in_0, w_1, w_0)
        tmp_1 = in_2.view(1, 19, -1, 64)
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        tmp_3 = tmp_0.view(1, 19, -1, 64)
        tmp_0 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = in_1.view(1, 19, 32, 64)
        tmp_6 = tmp_5.transpose(1, 2)
        tmp_5 = None
        tmp_7 = tmp_6.reshape(32, -1, 64)
        tmp_6 = None
        tmp_8 = tmp_2.reshape(32, -1, 64)
        tmp_2 = None
        tmp_9 = tmp_4.reshape(32, -1, 64)
        tmp_4 = None
        tmp_10 = tmp_8.transpose(1, 2)
        tmp_8 = None
        return (tmp_7, tmp_9, tmp_10)
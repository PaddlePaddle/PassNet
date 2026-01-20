import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.nn.functional.linear(in_0, w_1, w_0)
        tmp_1 = tmp_0.view(1, 18, 16, 3, 96)
        tmp_0 = None
        tmp_2 = tmp_1[Ellipsis, 0, slice(None, None, None)]
        tmp_3 = tmp_2.transpose(1, 2)
        tmp_2 = None
        tmp_4 = tmp_1[Ellipsis, 1, slice(None, None, None)]
        tmp_5 = tmp_4.transpose(1, 2)
        tmp_4 = None
        tmp_6 = tmp_1[Ellipsis, 2, slice(None, None, None)]
        tmp_1 = None
        tmp_7 = tmp_6.transpose(1, 2)
        tmp_6 = None
        tmp_8 = tmp_3.reshape(16, -1, 96)
        tmp_3 = None
        tmp_9 = tmp_5.reshape(16, -1, 96)
        tmp_5 = None
        tmp_10 = tmp_9.transpose(-1, -2)
        tmp_9 = None
        tmp_11 = tmp_7.reshape(16, -1, 96)
        tmp_7 = None
        return (tmp_8, tmp_10, tmp_11)
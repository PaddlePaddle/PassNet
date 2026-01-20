import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(1, 18, 16, 3, 96)
        tmp_1 = tmp_0[Ellipsis, 0, slice(None, None, None)]
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        tmp_3 = tmp_0[Ellipsis, 1, slice(None, None, None)]
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = tmp_0[Ellipsis, 2, slice(None, None, None)]
        tmp_0 = None
        tmp_6 = tmp_5.transpose(1, 2)
        tmp_5 = None
        tmp_7 = tmp_2.reshape(16, -1, 96)
        tmp_2 = None
        tmp_8 = tmp_4.reshape(16, -1, 96)
        tmp_4 = None
        tmp_9 = tmp_8.transpose(-1, -2)
        tmp_8 = None
        tmp_10 = tmp_6.reshape(16, -1, 96)
        tmp_6 = None
        return (tmp_10, tmp_7, tmp_9)
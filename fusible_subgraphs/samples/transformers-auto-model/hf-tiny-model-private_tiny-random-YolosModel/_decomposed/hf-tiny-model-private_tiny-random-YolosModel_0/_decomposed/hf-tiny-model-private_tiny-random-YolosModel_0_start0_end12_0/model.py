import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4):
        tmp_0 = torch.conv2d(in_0, w_1, w_0, (2, 2), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0.flatten(2)
        tmp_0 = None
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        tmp_3 = w_2.expand(1, -1, -1)
        tmp_4 = w_3.expand(1, -1, -1)
        tmp_5 = torch.cat((tmp_3, tmp_2, tmp_4), dim=1)
        tmp_3 = tmp_2 = tmp_4 = None
        tmp_6 = w_4[slice(None, None, None), 0, slice(None, None, None)]
        tmp_7 = tmp_6[slice(None, None, None), None]
        tmp_6 = None
        tmp_8 = w_4[slice(None, None, None), slice(-10, None, None), slice(None, None, None)]
        tmp_9 = w_4[slice(None, None, None), slice(1, -10, None), slice(None, None, None)]
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = tmp_10.view(1, 32, 15, 15)
        tmp_10 = None
        return (tmp_5, tmp_7, tmp_8, tmp_11)
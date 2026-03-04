import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.conv2d(in_4, tmp_1, tmp_0, (14, 14), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_5 = tmp_4.flatten(2)
        tmp_4 = None
        tmp_6 = tmp_5.transpose(1, 2)
        tmp_5 = None
        tmp_7 = tmp_2.expand(1, -1, -1)
        tmp_2 = None
        tmp_8 = torch.cat((tmp_7, tmp_6), dim=1)
        tmp_7 = tmp_6 = None
        tmp_9 = tmp_3[slice(None, None, None), slice(None, 1, None)]
        tmp_10 = tmp_3[slice(None, None, None), slice(1, None, None)]
        tmp_3 = None
        tmp_11 = tmp_10.reshape(1, 37, 37, 384)
        tmp_10 = None
        tmp_12 = tmp_11.permute(0, 3, 1, 2)
        tmp_11 = None
        tmp_13 = tmp_12.to(torch.float32)
        tmp_12 = None
        return (tmp_9, tmp_8, tmp_13)
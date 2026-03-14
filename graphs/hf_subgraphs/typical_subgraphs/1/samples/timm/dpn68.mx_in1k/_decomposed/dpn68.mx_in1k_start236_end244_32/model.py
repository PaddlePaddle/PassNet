import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = torch.nn.functional.relu(in_7, inplace=True)
        tmp_6 = torch.conv2d(tmp_5, tmp_4, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_5 = tmp_4 = None
        tmp_7 = tmp_6[slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, None, None)]
        tmp_8 = tmp_6[slice(None, None, None), slice(256, None, None), slice(None, None, None), slice(None, None, None)]
        tmp_6 = None
        tmp_9 = in_6 + tmp_7
        tmp_7 = None
        tmp_10 = torch.cat([in_5, tmp_8], dim=1)
        tmp_8 = None
        tmp_11 = torch.cat((tmp_9, tmp_10), dim=1)
        tmp_12 = torch.nn.functional.batch_norm(tmp_11, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 0.001)
        tmp_11 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        return (tmp_10, tmp_9, tmp_12)
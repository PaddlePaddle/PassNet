import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.relu(in_5, inplace=True)
        tmp_5 = torch.cat([in_4, tmp_4], dim=1)
        tmp_4 = None
        tmp_6 = tmp_5[slice(None, None, None), slice(None, 120, None), slice(None, None, None), slice(None, None, None)]
        tmp_5 = None
        tmp_7 = tmp_6.mean((2, 3), keepdim=True)
        tmp_8 = torch.conv2d(tmp_7, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_7 = tmp_3 = tmp_2 = None
        tmp_9 = torch.nn.functional.relu(tmp_8, inplace=True)
        tmp_8 = None
        tmp_10 = torch.conv2d(tmp_9, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_9 = tmp_1 = tmp_0 = None
        tmp_11 = torch.nn.functional.hardsigmoid(tmp_10, False)
        tmp_10 = None
        tmp_12 = tmp_6 * tmp_11
        tmp_6 = tmp_11 = None
        return (tmp_12,)
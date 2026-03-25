import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = in_8.view(128, 304, 192)
        tmp_9 = tmp_8.unsqueeze(1)
        tmp_8 = None
        tmp_10 = torch.conv2d(in_8, tmp_7, tmp_6, (1, 1), (0, 0), (1, 1), 1)
        tmp_7 = tmp_6 = None
        tmp_11 = tmp_10.view(128, 1, 192)
        tmp_10 = None
        tmp_12 = torch.nn.functional.softmax(tmp_11, 2, _stacklevel=5)
        tmp_11 = None
        tmp_13 = tmp_12.unsqueeze(-1)
        tmp_12 = None
        tmp_14 = torch.matmul(tmp_9, tmp_13)
        tmp_9 = tmp_13 = None
        tmp_15 = tmp_14.view(128, 304, 1, 1)
        tmp_14 = None
        tmp_16 = torch.conv2d(tmp_15, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_15 = tmp_1 = tmp_0 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (19, 1, 1), tmp_3, tmp_2, 1e-05)
        tmp_16 = tmp_3 = tmp_2 = None
        tmp_18 = torch.nn.functional.relu(tmp_17, inplace=True)
        tmp_17 = None
        tmp_19 = torch.conv2d(tmp_18, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_18 = tmp_5 = tmp_4 = None
        tmp_20 = in_8 + tmp_19
        tmp_19 = None
        return (tmp_20,)
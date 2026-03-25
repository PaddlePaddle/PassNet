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
        tmp_8 = torch.nn.functional.relu(in_8, inplace=True)
        tmp_9 = tmp_8.view(64, 512, 4096)
        tmp_10 = tmp_9.unsqueeze(1)
        tmp_9 = None
        tmp_11 = torch.conv2d(tmp_8, tmp_7, tmp_6, (1, 1), (0, 0), (1, 1), 1)
        tmp_7 = tmp_6 = None
        tmp_12 = tmp_11.view(64, 1, 4096)
        tmp_11 = None
        tmp_13 = torch.nn.functional.softmax(tmp_12, 2, _stacklevel=5)
        tmp_12 = None
        tmp_14 = tmp_13.unsqueeze(-1)
        tmp_13 = None
        tmp_15 = torch.matmul(tmp_10, tmp_14)
        tmp_10 = tmp_14 = None
        tmp_16 = tmp_15.view(64, 512, 1, 1)
        tmp_15 = None
        tmp_17 = torch.conv2d(tmp_16, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_16 = tmp_1 = tmp_0 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (128, 1, 1), tmp_3, tmp_2, 1e-05)
        tmp_17 = tmp_3 = tmp_2 = None
        tmp_19 = torch.nn.functional.relu(tmp_18, inplace=True)
        tmp_18 = None
        tmp_20 = torch.conv2d(tmp_19, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_19 = tmp_5 = tmp_4 = None
        tmp_21 = tmp_8 + tmp_20
        tmp_8 = tmp_20 = None
        return (tmp_21,)
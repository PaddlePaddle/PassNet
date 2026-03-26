import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = in_0.view(1, 80, 3072)
        tmp_9 = tmp_8.unsqueeze(1)
        tmp_8 = None
        tmp_10 = torch.conv2d(in_0, tmp_7, tmp_6, (1, 1), (0, 0), (1, 1), 1)
        tmp_7 = tmp_6 = None
        tmp_11 = tmp_10.view(1, 1, 3072)
        tmp_10 = None
        tmp_12 = torch.nn.functional.softmax(tmp_11, 2, _stacklevel=5)
        tmp_11 = None
        tmp_13 = tmp_12.unsqueeze(-1)
        tmp_12 = None
        tmp_14 = torch.matmul(tmp_9, tmp_13)
        tmp_9 = tmp_13 = None
        tmp_15 = tmp_14.view(1, 80, 1, 1)
        tmp_14 = None
        tmp_16 = torch.conv2d(tmp_15, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_15 = tmp_1 = tmp_0 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (16, 1, 1), tmp_3, tmp_2, 1e-05)
        tmp_16 = tmp_3 = tmp_2 = None
        tmp_18 = torch.nn.functional.relu(tmp_17, inplace=True)
        tmp_17 = None
        tmp_19 = torch.conv2d(tmp_18, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_18 = tmp_5 = tmp_4 = None
        tmp_20 = in_0 + tmp_19
        tmp_19 = None
        return (tmp_20,)
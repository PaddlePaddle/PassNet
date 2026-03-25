import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_7 = torch.conv2d(tmp_6, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_0 = None
        tmp_8 = torch.conv2d(tmp_6, tmp_1, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_6 = tmp_1 = None
        tmp_9 = in_1 + tmp_7
        tmp_7 = None
        tmp_10 = torch.cat([in_2, tmp_8], dim=1)
        tmp_8 = None
        tmp_11 = torch.cat((tmp_9, tmp_10), dim=1)
        tmp_12 = torch.nn.functional.batch_norm(tmp_11, tmp_2, tmp_3, tmp_5, tmp_4, False, 0.1, 0.001)
        tmp_11 = tmp_2 = tmp_3 = tmp_5 = tmp_4 = None
        return (tmp_10, tmp_9, tmp_12)
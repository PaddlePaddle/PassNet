import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_6 = torch.conv2d(tmp_5, tmp_4, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_5 = tmp_4 = None
        tmp_7 = torch.cat([in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, tmp_6], 1)
        tmp_8 = torch.nn.functional.batch_norm(tmp_7, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_7 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_9 = torch.nn.functional.relu(tmp_8, inplace=True)
        tmp_8 = None
        return (tmp_6, tmp_9)
import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = torch.conv2d(in_8, tmp_4, None, (2, 2), (3, 3), (1, 1), 60)
        tmp_4 = None
        tmp_7 = torch.conv2d(in_9, tmp_5, None, (2, 2), (4, 4), (1, 1), 60)
        tmp_5 = None
        tmp_8 = torch.cat([in_6, in_7, tmp_6, tmp_7], 1)
        tmp_6 = tmp_7 = None
        tmp_9 = torch.nn.functional.batch_norm(tmp_8, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_8 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_10 = torch.nn.functional.silu(tmp_9, inplace=True)
        tmp_9 = None
        tmp_11 = tmp_10.mean((2, 3), keepdim=True)
        return (tmp_10, tmp_11)
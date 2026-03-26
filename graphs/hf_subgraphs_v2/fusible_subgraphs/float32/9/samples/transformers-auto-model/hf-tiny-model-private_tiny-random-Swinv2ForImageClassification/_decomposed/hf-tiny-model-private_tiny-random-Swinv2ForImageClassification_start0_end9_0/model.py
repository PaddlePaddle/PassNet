import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = torch.conv2d(tmp_0, tmp_4, tmp_3, (2, 2), (0, 0), (1, 1), 1)
        tmp_0 = tmp_4 = tmp_3 = None
        tmp_6 = tmp_5.flatten(2)
        tmp_5 = None
        tmp_7 = tmp_6.transpose(1, 2)
        tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (16,), tmp_2, tmp_1, 1e-05)
        tmp_7 = tmp_2 = tmp_1 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
        tmp_8 = None
        tmp_10 = tmp_9.view(1, 16, 16, 16)
        tmp_11 = torch.nn.functional.pad(tmp_10, (0, 0, 0, 0, 0, 0), 'constant', None)
        tmp_10 = None
        tmp_12 = tmp_11.view(1, 8, 2, 8, 2, 16)
        tmp_11 = None
        tmp_13 = tmp_12.permute(0, 1, 3, 2, 4, 5)
        tmp_12 = None
        return (tmp_9, tmp_13)
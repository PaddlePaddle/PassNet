import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = in_0
        tmp_8 = torch.conv2d(tmp_7, tmp_5, tmp_4, (16, 16), (0, 0), (1, 1), 1)
        tmp_7 = tmp_5 = tmp_4 = None
        tmp_9 = tmp_8.flatten(2)
        tmp_8 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = tmp_10 + tmp_6
        tmp_10 = tmp_6 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
        tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), tmp_3, tmp_2, 1e-06)
        tmp_3 = tmp_2 = None
        tmp_14 = torch.nn.functional.linear(tmp_13, tmp_1, tmp_0)
        tmp_13 = tmp_1 = tmp_0 = None
        return (tmp_14, tmp_12)
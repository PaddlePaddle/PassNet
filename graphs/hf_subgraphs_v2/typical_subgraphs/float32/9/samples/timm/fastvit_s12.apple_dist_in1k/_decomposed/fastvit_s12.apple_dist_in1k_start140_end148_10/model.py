import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = torch.conv2d(in_1, tmp_2, tmp_1, (1, 1), (0, 0), (1, 1), 1)
        tmp_2 = tmp_1 = None
        tmp_10 = torch.nn.functional.gelu(tmp_9, approximate='none')
        tmp_9 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False)
        tmp_10 = None
        tmp_12 = torch.conv2d(tmp_11, tmp_4, tmp_3, (1, 1), (0, 0), (1, 1), 1)
        tmp_11 = tmp_4 = tmp_3 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False)
        tmp_12 = None
        tmp_14 = tmp_13 * tmp_0
        tmp_13 = tmp_0 = None
        tmp_15 = in_0 + tmp_14
        tmp_14 = None
        tmp_16 = torch.nn.functional.batch_norm(tmp_15, tmp_5, tmp_6, tmp_8, tmp_7, False, 0.1, 1e-05)
        tmp_5 = tmp_6 = tmp_8 = tmp_7 = None
        return (tmp_16, tmp_15)
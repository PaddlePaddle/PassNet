import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = w_9
        tmp_10 = w_10
        tmp_11 = w_11
        tmp_12 = w_12
        tmp_13 = w_13
        tmp_14 = w_14
        tmp_15 = w_15
        tmp_16 = torch.nn.functional.gelu(in_0, approximate='none')
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.0, False, False)
        tmp_16 = None
        tmp_18 = torch.nn.functional.linear(tmp_17, tmp_1, tmp_0)
        tmp_17 = tmp_1 = tmp_0 = None
        tmp_19 = torch.nn.functional.dropout(tmp_18, 0.0, False, False)
        tmp_18 = None
        tmp_20 = in_1 + tmp_19
        tmp_19 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (384,), tmp_3, tmp_2, 1e-06)
        tmp_20 = tmp_3 = tmp_2 = None
        tmp_22 = tmp_21[slice(None, None, None), slice(0, None, None)]
        tmp_21 = None
        tmp_23 = tmp_22.reshape(1, 16, 12, -1)
        tmp_22 = None
        tmp_24 = tmp_23.permute(0, 3, 1, 2)
        tmp_23 = None
        tmp_25 = torch.conv_transpose2d(tmp_24, tmp_4, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_24 = tmp_4 = None
        tmp_26 = torch.nn.functional.batch_norm(tmp_25, tmp_5, tmp_6, tmp_8, tmp_7, False, 0.1, 1e-05)
        tmp_25 = tmp_5 = tmp_6 = tmp_8 = tmp_7 = None
        tmp_27 = torch.nn.functional.relu(tmp_26, inplace=True)
        tmp_26 = None
        tmp_28 = torch.conv_transpose2d(tmp_27, tmp_9, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_27 = tmp_9 = None
        tmp_29 = torch.nn.functional.batch_norm(tmp_28, tmp_10, tmp_11, tmp_13, tmp_12, False, 0.1, 1e-05)
        tmp_28 = tmp_10 = tmp_11 = tmp_13 = tmp_12 = None
        tmp_30 = torch.nn.functional.relu(tmp_29, inplace=True)
        tmp_29 = None
        tmp_31 = torch.conv2d(tmp_30, tmp_15, tmp_14, (1, 1), (0, 0), (1, 1), 1)
        tmp_30 = tmp_15 = tmp_14 = None
        return (tmp_31,)
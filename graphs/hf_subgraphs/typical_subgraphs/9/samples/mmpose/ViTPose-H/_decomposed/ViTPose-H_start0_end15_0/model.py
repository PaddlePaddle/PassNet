import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = w_6
        tmp_8 = torch.conv2d(tmp_0, tmp_6, tmp_5, (16, 16), (2, 2), (1, 1), 1)
        tmp_0 = tmp_6 = tmp_5 = None
        tmp_9 = tmp_8.flatten(2)
        tmp_8 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = tmp_10 + tmp_7
        tmp_10 = tmp_7 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
        tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (1280,), tmp_4, tmp_3, 1e-06)
        tmp_4 = tmp_3 = None
        tmp_14 = torch.nn.functional.linear(tmp_13, tmp_2, tmp_1)
        tmp_13 = tmp_2 = tmp_1 = None
        tmp_15 = tmp_14.reshape(1, 192, 3, 16, 80)
        tmp_14 = None
        tmp_16 = tmp_15.permute(2, 0, 3, 1, 4)
        tmp_15 = None
        tmp_17 = tmp_16[0]
        tmp_18 = tmp_16[1]
        tmp_19 = tmp_16[2]
        tmp_16 = None
        tmp_20 = torch.nn.functional.scaled_dot_product_attention(tmp_17, tmp_18, tmp_19, dropout_p=0.0)
        tmp_17 = tmp_18 = tmp_19 = None
        tmp_21 = tmp_20.transpose(1, 2)
        tmp_20 = None
        tmp_22 = tmp_21.reshape(1, 192, 1280)
        tmp_21 = None
        return (tmp_12, tmp_22)
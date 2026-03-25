import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0):
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
        tmp_10 = in_0
        tmp_11 = torch.conv2d(tmp_10, tmp_7, tmp_6, (14, 14), (0, 0), (1, 1), 1)
        tmp_10 = tmp_7 = tmp_6 = None
        tmp_12 = tmp_11.flatten(2)
        tmp_11 = None
        tmp_13 = tmp_12.transpose(1, 2)
        tmp_12 = None
        tmp_14 = tmp_8.expand(1, -1, -1)
        tmp_8 = None
        tmp_15 = torch.cat([tmp_14, tmp_13], dim=1)
        tmp_14 = tmp_13 = None
        tmp_16 = tmp_15 + tmp_9
        tmp_15 = tmp_9 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.0, False, False)
        tmp_16 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (1408,), tmp_5, tmp_4, 1e-06)
        tmp_5 = tmp_4 = None
        tmp_19 = torch.cat((tmp_2, tmp_0, tmp_3))
        tmp_2 = tmp_0 = tmp_3 = None
        tmp_20 = torch.nn.functional.linear(tmp_18, weight=tmp_1, bias=tmp_19)
        tmp_18 = tmp_1 = tmp_19 = None
        tmp_21 = tmp_20.reshape(1, 257, 3, 16, -1)
        tmp_20 = None
        tmp_22 = tmp_21.permute(2, 0, 3, 1, 4)
        tmp_21 = None
        tmp_23 = tmp_22.unbind(0)
        tmp_22 = None
        tmp_24 = tmp_23[0]
        tmp_25 = tmp_23[1]
        tmp_26 = tmp_23[2]
        tmp_23 = None
        tmp_27 = torch.nn.functional.scaled_dot_product_attention(tmp_24, tmp_25, tmp_26, attn_mask=None, dropout_p=0.0)
        tmp_24 = tmp_25 = tmp_26 = None
        tmp_28 = tmp_27.transpose(1, 2)
        tmp_27 = None
        tmp_29 = tmp_28.reshape(1, 257, 1408)
        tmp_28 = None
        return (tmp_17, tmp_29)
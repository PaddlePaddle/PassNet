import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0):
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
        tmp_12 = in_0
        tmp_13 = torch.nn.functional.interpolate(tmp_12, size=(240, 240), mode='bicubic', align_corners=False)
        tmp_14 = torch.conv2d(tmp_13, tmp_5, tmp_4, (12, 12), (0, 0), (1, 1), 1)
        tmp_13 = tmp_5 = tmp_4 = None
        tmp_15 = tmp_14.flatten(2)
        tmp_14 = None
        tmp_16 = tmp_15.transpose(1, 2)
        tmp_15 = None
        tmp_17 = tmp_8.expand(1, -1, -1)
        tmp_8 = None
        tmp_18 = torch.cat((tmp_17, tmp_16), dim=1)
        tmp_17 = tmp_16 = None
        tmp_19 = tmp_18 + tmp_10
        tmp_18 = tmp_10 = None
        tmp_20 = torch.nn.functional.dropout(tmp_19, 0.0, False, False)
        tmp_19 = None
        tmp_21 = torch.conv2d(tmp_12, tmp_7, tmp_6, (16, 16), (0, 0), (1, 1), 1)
        tmp_12 = tmp_7 = tmp_6 = None
        tmp_22 = tmp_21.flatten(2)
        tmp_21 = None
        tmp_23 = tmp_22.transpose(1, 2)
        tmp_22 = None
        tmp_24 = tmp_9.expand(1, -1, -1)
        tmp_9 = None
        tmp_25 = torch.cat((tmp_24, tmp_23), dim=1)
        tmp_24 = tmp_23 = None
        tmp_26 = tmp_25 + tmp_11
        tmp_25 = tmp_11 = None
        tmp_27 = torch.nn.functional.dropout(tmp_26, 0.0, False, False)
        tmp_26 = None
        tmp_28 = torch.nn.functional.layer_norm(tmp_20, (128,), tmp_3, tmp_2, 1e-06)
        tmp_3 = tmp_2 = None
        tmp_29 = torch.nn.functional.linear(tmp_28, tmp_1, tmp_0)
        tmp_28 = tmp_1 = tmp_0 = None
        tmp_30 = tmp_29.reshape(1, 401, 3, 4, 32)
        tmp_29 = None
        tmp_31 = tmp_30.permute(2, 0, 3, 1, 4)
        tmp_30 = None
        tmp_32 = tmp_31.unbind(0)
        tmp_31 = None
        tmp_33 = tmp_32[0]
        tmp_34 = tmp_32[1]
        tmp_35 = tmp_32[2]
        tmp_32 = None
        tmp_36 = torch.nn.functional.scaled_dot_product_attention(tmp_33, tmp_34, tmp_35, attn_mask=None, dropout_p=0.0)
        tmp_33 = tmp_34 = tmp_35 = None
        tmp_37 = tmp_36.transpose(1, 2)
        tmp_36 = None
        tmp_38 = tmp_37.reshape(1, 401, 128)
        tmp_37 = None
        return (tmp_38, tmp_20, tmp_27)
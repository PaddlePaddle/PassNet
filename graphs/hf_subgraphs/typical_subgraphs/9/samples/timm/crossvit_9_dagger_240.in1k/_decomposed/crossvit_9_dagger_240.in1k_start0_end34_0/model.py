import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, in_0):
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
        tmp_16 = w_16
        tmp_17 = w_17
        tmp_18 = w_18
        tmp_19 = w_19
        tmp_20 = in_0
        tmp_21 = torch.nn.functional.interpolate(tmp_20, size=(240, 240), mode='bicubic', align_corners=False)
        tmp_22 = torch.conv2d(tmp_21, tmp_5, tmp_4, (4, 4), (3, 3), (1, 1), 1)
        tmp_21 = tmp_5 = tmp_4 = None
        tmp_23 = torch.nn.functional.relu(tmp_22, inplace=True)
        tmp_22 = None
        tmp_24 = torch.conv2d(tmp_23, tmp_7, tmp_6, (3, 3), (0, 0), (1, 1), 1)
        tmp_23 = tmp_7 = tmp_6 = None
        tmp_25 = torch.nn.functional.relu(tmp_24, inplace=True)
        tmp_24 = None
        tmp_26 = torch.conv2d(tmp_25, tmp_9, tmp_8, (1, 1), (1, 1), (1, 1), 1)
        tmp_25 = tmp_9 = tmp_8 = None
        tmp_27 = tmp_26.flatten(2)
        tmp_26 = None
        tmp_28 = tmp_27.transpose(1, 2)
        tmp_27 = None
        tmp_29 = tmp_16.expand(1, -1, -1)
        tmp_16 = None
        tmp_30 = torch.cat((tmp_29, tmp_28), dim=1)
        tmp_29 = tmp_28 = None
        tmp_31 = tmp_30 + tmp_18
        tmp_30 = tmp_18 = None
        tmp_32 = torch.nn.functional.dropout(tmp_31, 0.0, False, False)
        tmp_31 = None
        tmp_33 = torch.conv2d(tmp_20, tmp_11, tmp_10, (4, 4), (3, 3), (1, 1), 1)
        tmp_20 = tmp_11 = tmp_10 = None
        tmp_34 = torch.nn.functional.relu(tmp_33, inplace=True)
        tmp_33 = None
        tmp_35 = torch.conv2d(tmp_34, tmp_13, tmp_12, (2, 2), (1, 1), (1, 1), 1)
        tmp_34 = tmp_13 = tmp_12 = None
        tmp_36 = torch.nn.functional.relu(tmp_35, inplace=True)
        tmp_35 = None
        tmp_37 = torch.conv2d(tmp_36, tmp_15, tmp_14, (2, 2), (1, 1), (1, 1), 1)
        tmp_36 = tmp_15 = tmp_14 = None
        tmp_38 = tmp_37.flatten(2)
        tmp_37 = None
        tmp_39 = tmp_38.transpose(1, 2)
        tmp_38 = None
        tmp_40 = tmp_17.expand(1, -1, -1)
        tmp_17 = None
        tmp_41 = torch.cat((tmp_40, tmp_39), dim=1)
        tmp_40 = tmp_39 = None
        tmp_42 = tmp_41 + tmp_19
        tmp_41 = tmp_19 = None
        tmp_43 = torch.nn.functional.dropout(tmp_42, 0.0, False, False)
        tmp_42 = None
        tmp_44 = torch.nn.functional.layer_norm(tmp_32, (128,), tmp_3, tmp_2, 1e-06)
        tmp_3 = tmp_2 = None
        tmp_45 = torch.nn.functional.linear(tmp_44, tmp_1, tmp_0)
        tmp_44 = tmp_1 = tmp_0 = None
        tmp_46 = tmp_45.reshape(1, 401, 3, 4, 32)
        tmp_45 = None
        tmp_47 = tmp_46.permute(2, 0, 3, 1, 4)
        tmp_46 = None
        tmp_48 = tmp_47.unbind(0)
        tmp_47 = None
        tmp_49 = tmp_48[0]
        tmp_50 = tmp_48[1]
        tmp_51 = tmp_48[2]
        tmp_48 = None
        tmp_52 = torch.nn.functional.scaled_dot_product_attention(tmp_49, tmp_50, tmp_51, attn_mask=None, dropout_p=0.0)
        tmp_49 = tmp_50 = tmp_51 = None
        tmp_53 = tmp_52.transpose(1, 2)
        tmp_52 = None
        tmp_54 = tmp_53.reshape(1, 401, 128)
        tmp_53 = None
        return (tmp_54, tmp_32, tmp_43)
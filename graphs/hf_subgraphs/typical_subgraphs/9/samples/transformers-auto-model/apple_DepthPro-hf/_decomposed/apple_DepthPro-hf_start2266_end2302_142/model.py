import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, in_0, in_1, in_2, in_3, in_4):
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
        tmp_20 = w_20
        tmp_21 = w_21
        tmp_22 = w_22
        tmp_23 = w_23
        tmp_24 = torch.nn.functional.linear(in_2, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_25 = tmp_24.view(1, -1, 16, 64)
        tmp_24 = None
        tmp_26 = tmp_25.transpose(1, 2)
        tmp_25 = None
        tmp_27 = tmp_26.contiguous()
        tmp_26 = None
        tmp_28 = in_1.contiguous()
        tmp_29 = in_4.contiguous()
        tmp_30 = torch.nn.functional.scaled_dot_product_attention(tmp_27, tmp_28, tmp_29, attn_mask=None, dropout_p=0.0, scale=0.125, is_causal=False)
        tmp_27 = tmp_28 = tmp_29 = None
        tmp_31 = tmp_30.transpose(1, 2)
        tmp_30 = None
        tmp_32 = tmp_31.contiguous()
        tmp_31 = None
        tmp_33 = tmp_32.reshape((1, 577, 1024))
        tmp_32 = None
        tmp_34 = torch.nn.functional.linear(tmp_33, tmp_5, tmp_4)
        tmp_33 = tmp_5 = tmp_4 = None
        tmp_35 = torch.nn.functional.dropout(tmp_34, 0.0, False, False)
        tmp_34 = None
        tmp_36 = tmp_35 * tmp_6
        tmp_35 = tmp_6 = None
        tmp_37 = tmp_36 + in_3
        tmp_36 = None
        tmp_38 = torch.nn.functional.layer_norm(tmp_37, (1024,), tmp_13, tmp_12, 1e-06)
        tmp_13 = tmp_12 = None
        tmp_39 = torch.nn.functional.linear(tmp_38, tmp_9, tmp_8)
        tmp_38 = tmp_9 = tmp_8 = None
        tmp_40 = torch.nn.functional.gelu(tmp_39)
        tmp_39 = None
        tmp_41 = torch.nn.functional.linear(tmp_40, tmp_11, tmp_10)
        tmp_40 = tmp_11 = tmp_10 = None
        tmp_42 = tmp_41 * tmp_7
        tmp_41 = tmp_7 = None
        tmp_43 = tmp_42 + tmp_37
        tmp_42 = tmp_37 = None
        tmp_44 = torch.nn.functional.layer_norm(tmp_43, (1024,), tmp_15, tmp_14, 1e-06)
        tmp_43 = tmp_15 = tmp_14 = None
        tmp_45 = torch.nn.functional.linear(tmp_44, tmp_17, tmp_16)
        tmp_44 = tmp_17 = tmp_16 = None
        tmp_46 = tmp_45[slice(None, None, None), slice(-576, None, None), slice(None, None, None)]
        tmp_45 = None
        tmp_47 = tmp_46.reshape(1, 24, 24, 128)
        tmp_46 = None
        tmp_48 = tmp_47.permute(0, 3, 1, 2)
        tmp_47 = None
        tmp_49 = torch.nn.functional.interpolate(tmp_48, size=(24, 24), mode='bilinear', align_corners=False)
        tmp_48 = None
        tmp_50 = torch.conv2d(in_0, tmp_1, tmp_0, (2, 2), (1, 1), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_51 = torch.nn.functional.relu(tmp_50, inplace=True)
        tmp_50 = None
        tmp_52 = tmp_49 + tmp_51
        tmp_49 = tmp_51 = None
        tmp_53 = torch.nn.functional.interpolate(tmp_52, size=(24, 24), mode='bilinear', align_corners=False)
        tmp_52 = None
        tmp_54 = torch.conv2d(tmp_53, tmp_19, tmp_18, (2, 2), (1, 1), (1, 1), 1)
        tmp_53 = tmp_19 = tmp_18 = None
        tmp_55 = torch.nn.functional.relu(tmp_54, inplace=True)
        tmp_54 = None
        tmp_56 = torch.conv2d(tmp_55, tmp_21, tmp_20, (2, 2), (1, 1), (1, 1), 1)
        tmp_55 = tmp_21 = tmp_20 = None
        tmp_57 = torch.nn.functional.relu(tmp_56, inplace=True)
        tmp_56 = None
        tmp_58 = torch.conv2d(tmp_57, tmp_23, tmp_22, (1, 1), (0, 0), (1, 1), 1)
        tmp_57 = tmp_23 = tmp_22 = None
        tmp_59 = tmp_58.flatten()
        tmp_58 = None
        return (tmp_59,)
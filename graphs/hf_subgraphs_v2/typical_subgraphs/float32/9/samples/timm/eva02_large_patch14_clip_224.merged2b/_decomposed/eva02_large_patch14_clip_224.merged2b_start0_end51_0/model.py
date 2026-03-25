import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, in_0):
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
        tmp_14 = in_0
        tmp_15 = torch.conv2d(tmp_14, tmp_10, tmp_9, (14, 14), (0, 0), (1, 1), 1)
        tmp_14 = tmp_10 = tmp_9 = None
        tmp_16 = tmp_15.flatten(2)
        tmp_15 = None
        tmp_17 = tmp_16.transpose(1, 2)
        tmp_16 = None
        tmp_18 = tmp_12.expand(1, -1, -1)
        tmp_12 = None
        tmp_19 = torch.cat([tmp_18, tmp_17], dim=1)
        tmp_18 = tmp_17 = None
        tmp_20 = tmp_19 + tmp_13
        tmp_19 = tmp_13 = None
        tmp_21 = torch.nn.functional.dropout(tmp_20, 0.0, False, False)
        tmp_20 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (1024,), tmp_8, tmp_7, 1e-06)
        tmp_8 = tmp_7 = None
        tmp_23 = torch.nn.functional.linear(tmp_22, tmp_4, tmp_3)
        tmp_4 = tmp_3 = None
        tmp_24 = tmp_23.reshape(1, 257, 16, -1)
        tmp_23 = None
        tmp_25 = tmp_24.transpose(1, 2)
        tmp_24 = None
        tmp_26 = torch.nn.functional.linear(tmp_22, tmp_0, None)
        tmp_0 = None
        tmp_27 = tmp_26.reshape(1, 257, 16, -1)
        tmp_26 = None
        tmp_28 = tmp_27.transpose(1, 2)
        tmp_27 = None
        tmp_29 = torch.nn.functional.linear(tmp_22, tmp_6, tmp_5)
        tmp_22 = tmp_6 = tmp_5 = None
        tmp_30 = tmp_29.reshape(1, 257, 16, -1)
        tmp_29 = None
        tmp_31 = tmp_30.transpose(1, 2)
        tmp_30 = None
        tmp_32 = tmp_25[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
        tmp_33 = tmp_25[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_25 = None
        tmp_34 = tmp_11.tensor_split(2, -1)
        tmp_35 = tmp_34[0]
        tmp_36 = tmp_34[1]
        tmp_34 = None
        tmp_37 = tmp_33 * tmp_36
        tmp_36 = None
        tmp_38 = tmp_33[Ellipsis, slice(1, None, 2)]
        tmp_39 = -tmp_38
        tmp_38 = None
        tmp_40 = tmp_33[Ellipsis, slice(None, None, 2)]
        tmp_33 = None
        tmp_41 = torch.stack([tmp_39, tmp_40], -1)
        tmp_39 = tmp_40 = None
        tmp_42 = tmp_41.reshape((1, 16, 256, 64))
        tmp_41 = None
        tmp_43 = tmp_42 * tmp_35
        tmp_42 = tmp_35 = None
        tmp_44 = tmp_37 + tmp_43
        tmp_37 = tmp_43 = None
        tmp_45 = torch.cat([tmp_32, tmp_44], dim=2)
        tmp_32 = tmp_44 = None
        tmp_46 = tmp_45.type_as(tmp_31)
        tmp_45 = None
        tmp_47 = tmp_28[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
        tmp_48 = tmp_28[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_28 = None
        tmp_49 = tmp_11.tensor_split(2, -1)
        tmp_11 = None
        tmp_50 = tmp_49[0]
        tmp_51 = tmp_49[1]
        tmp_49 = None
        tmp_52 = tmp_48 * tmp_51
        tmp_51 = None
        tmp_53 = tmp_48[Ellipsis, slice(1, None, 2)]
        tmp_54 = -tmp_53
        tmp_53 = None
        tmp_55 = tmp_48[Ellipsis, slice(None, None, 2)]
        tmp_48 = None
        tmp_56 = torch.stack([tmp_54, tmp_55], -1)
        tmp_54 = tmp_55 = None
        tmp_57 = tmp_56.reshape((1, 16, 256, 64))
        tmp_56 = None
        tmp_58 = tmp_57 * tmp_50
        tmp_57 = tmp_50 = None
        tmp_59 = tmp_52 + tmp_58
        tmp_52 = tmp_58 = None
        tmp_60 = torch.cat([tmp_47, tmp_59], dim=2)
        tmp_47 = tmp_59 = None
        tmp_61 = tmp_60.type_as(tmp_31)
        tmp_60 = None
        tmp_62 = torch.nn.functional.scaled_dot_product_attention(tmp_46, tmp_61, tmp_31, attn_mask=None, dropout_p=0.0)
        tmp_46 = tmp_61 = tmp_31 = None
        tmp_63 = tmp_62.transpose(1, 2)
        tmp_62 = None
        tmp_64 = tmp_63.reshape(1, 257, 1024)
        tmp_63 = None
        tmp_65 = torch.nn.functional.layer_norm(tmp_64, (1024,), tmp_2, tmp_1, 1e-06)
        tmp_64 = tmp_2 = tmp_1 = None
        return (tmp_21, tmp_65)
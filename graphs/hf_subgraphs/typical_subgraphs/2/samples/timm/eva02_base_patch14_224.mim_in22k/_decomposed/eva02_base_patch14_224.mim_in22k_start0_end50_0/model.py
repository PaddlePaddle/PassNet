import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = in_8
        tmp_9 = in_9
        tmp_10 = in_10
        tmp_11 = in_11
        tmp_12 = in_12
        tmp_13 = torch.conv2d(tmp_12, tmp_8, tmp_7, (14, 14), (0, 0), (1, 1), 1)
        tmp_12 = tmp_8 = tmp_7 = None
        tmp_14 = tmp_13.flatten(2)
        tmp_13 = None
        tmp_15 = tmp_14.transpose(1, 2)
        tmp_14 = None
        tmp_16 = tmp_10.expand(1, -1, -1)
        tmp_10 = None
        tmp_17 = torch.cat([tmp_16, tmp_15], dim=1)
        tmp_16 = tmp_15 = None
        tmp_18 = tmp_17 + tmp_11
        tmp_17 = tmp_11 = None
        tmp_19 = torch.nn.functional.dropout(tmp_18, 0.0, False, False)
        tmp_18 = None
        tmp_20 = torch.nn.functional.layer_norm(tmp_19, (768,), tmp_6, tmp_5, 1e-06)
        tmp_6 = tmp_5 = None
        tmp_21 = torch.nn.functional.linear(tmp_20, tmp_2, tmp_1)
        tmp_2 = tmp_1 = None
        tmp_22 = tmp_21.reshape(1, 257, 12, -1)
        tmp_21 = None
        tmp_23 = tmp_22.transpose(1, 2)
        tmp_22 = None
        tmp_24 = torch.nn.functional.linear(tmp_20, tmp_0, None)
        tmp_0 = None
        tmp_25 = tmp_24.reshape(1, 257, 12, -1)
        tmp_24 = None
        tmp_26 = tmp_25.transpose(1, 2)
        tmp_25 = None
        tmp_27 = torch.nn.functional.linear(tmp_20, tmp_4, tmp_3)
        tmp_20 = tmp_4 = tmp_3 = None
        tmp_28 = tmp_27.reshape(1, 257, 12, -1)
        tmp_27 = None
        tmp_29 = tmp_28.transpose(1, 2)
        tmp_28 = None
        tmp_30 = tmp_23[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
        tmp_31 = tmp_23[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_23 = None
        tmp_32 = tmp_9.tensor_split(2, -1)
        tmp_33 = tmp_32[0]
        tmp_34 = tmp_32[1]
        tmp_32 = None
        tmp_35 = tmp_31 * tmp_34
        tmp_34 = None
        tmp_36 = tmp_31[Ellipsis, slice(1, None, 2)]
        tmp_37 = -tmp_36
        tmp_36 = None
        tmp_38 = tmp_31[Ellipsis, slice(None, None, 2)]
        tmp_31 = None
        tmp_39 = torch.stack([tmp_37, tmp_38], -1)
        tmp_37 = tmp_38 = None
        tmp_40 = tmp_39.reshape((1, 12, 256, 64))
        tmp_39 = None
        tmp_41 = tmp_40 * tmp_33
        tmp_40 = tmp_33 = None
        tmp_42 = tmp_35 + tmp_41
        tmp_35 = tmp_41 = None
        tmp_43 = torch.cat([tmp_30, tmp_42], dim=2)
        tmp_30 = tmp_42 = None
        tmp_44 = tmp_43.type_as(tmp_29)
        tmp_43 = None
        tmp_45 = tmp_26[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
        tmp_46 = tmp_26[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_26 = None
        tmp_47 = tmp_9.tensor_split(2, -1)
        tmp_9 = None
        tmp_48 = tmp_47[0]
        tmp_49 = tmp_47[1]
        tmp_47 = None
        tmp_50 = tmp_46 * tmp_49
        tmp_49 = None
        tmp_51 = tmp_46[Ellipsis, slice(1, None, 2)]
        tmp_52 = -tmp_51
        tmp_51 = None
        tmp_53 = tmp_46[Ellipsis, slice(None, None, 2)]
        tmp_46 = None
        tmp_54 = torch.stack([tmp_52, tmp_53], -1)
        tmp_52 = tmp_53 = None
        tmp_55 = tmp_54.reshape((1, 12, 256, 64))
        tmp_54 = None
        tmp_56 = tmp_55 * tmp_48
        tmp_55 = tmp_48 = None
        tmp_57 = tmp_50 + tmp_56
        tmp_50 = tmp_56 = None
        tmp_58 = torch.cat([tmp_45, tmp_57], dim=2)
        tmp_45 = tmp_57 = None
        tmp_59 = tmp_58.type_as(tmp_29)
        tmp_58 = None
        tmp_60 = torch.nn.functional.scaled_dot_product_attention(tmp_44, tmp_59, tmp_29, attn_mask=None, dropout_p=0.0)
        tmp_44 = tmp_59 = tmp_29 = None
        tmp_61 = tmp_60.transpose(1, 2)
        tmp_60 = None
        tmp_62 = tmp_61.reshape(1, 257, 768)
        tmp_61 = None
        return (tmp_19, tmp_62)
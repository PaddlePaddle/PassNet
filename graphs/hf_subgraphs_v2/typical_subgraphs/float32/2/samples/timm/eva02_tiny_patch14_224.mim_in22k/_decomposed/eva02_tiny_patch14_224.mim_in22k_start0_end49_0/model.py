import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
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
        tmp_12 = torch.conv2d(tmp_11, tmp_7, tmp_6, (14, 14), (0, 0), (1, 1), 1)
        tmp_11 = tmp_7 = tmp_6 = None
        tmp_13 = tmp_12.flatten(2)
        tmp_12 = None
        tmp_14 = tmp_13.transpose(1, 2)
        tmp_13 = None
        tmp_15 = tmp_9.expand(1, -1, -1)
        tmp_9 = None
        tmp_16 = torch.cat([tmp_15, tmp_14], dim=1)
        tmp_15 = tmp_14 = None
        tmp_17 = tmp_16 + tmp_10
        tmp_16 = tmp_10 = None
        tmp_18 = torch.nn.functional.dropout(tmp_17, 0.0, False, False)
        tmp_17 = None
        tmp_19 = torch.nn.functional.layer_norm(tmp_18, (192,), tmp_5, tmp_4, 1e-06)
        tmp_5 = tmp_4 = None
        tmp_20 = torch.cat((tmp_2, tmp_0, tmp_3))
        tmp_2 = tmp_0 = tmp_3 = None
        tmp_21 = torch.nn.functional.linear(tmp_19, weight=tmp_1, bias=tmp_20)
        tmp_19 = tmp_1 = tmp_20 = None
        tmp_22 = tmp_21.reshape(1, 257, 3, 3, -1)
        tmp_21 = None
        tmp_23 = tmp_22.permute(2, 0, 3, 1, 4)
        tmp_22 = None
        tmp_24 = tmp_23.unbind(0)
        tmp_23 = None
        tmp_25 = tmp_24[0]
        tmp_26 = tmp_24[1]
        tmp_27 = tmp_24[2]
        tmp_24 = None
        tmp_28 = tmp_25[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
        tmp_29 = tmp_25[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_25 = None
        tmp_30 = tmp_8.tensor_split(2, -1)
        tmp_31 = tmp_30[0]
        tmp_32 = tmp_30[1]
        tmp_30 = None
        tmp_33 = tmp_29 * tmp_32
        tmp_32 = None
        tmp_34 = tmp_29[Ellipsis, slice(1, None, 2)]
        tmp_35 = -tmp_34
        tmp_34 = None
        tmp_36 = tmp_29[Ellipsis, slice(None, None, 2)]
        tmp_29 = None
        tmp_37 = torch.stack([tmp_35, tmp_36], -1)
        tmp_35 = tmp_36 = None
        tmp_38 = tmp_37.reshape((1, 3, 256, 64))
        tmp_37 = None
        tmp_39 = tmp_38 * tmp_31
        tmp_38 = tmp_31 = None
        tmp_40 = tmp_33 + tmp_39
        tmp_33 = tmp_39 = None
        tmp_41 = torch.cat([tmp_28, tmp_40], dim=2)
        tmp_28 = tmp_40 = None
        tmp_42 = tmp_41.type_as(tmp_27)
        tmp_41 = None
        tmp_43 = tmp_26[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
        tmp_44 = tmp_26[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_26 = None
        tmp_45 = tmp_8.tensor_split(2, -1)
        tmp_8 = None
        tmp_46 = tmp_45[0]
        tmp_47 = tmp_45[1]
        tmp_45 = None
        tmp_48 = tmp_44 * tmp_47
        tmp_47 = None
        tmp_49 = tmp_44[Ellipsis, slice(1, None, 2)]
        tmp_50 = -tmp_49
        tmp_49 = None
        tmp_51 = tmp_44[Ellipsis, slice(None, None, 2)]
        tmp_44 = None
        tmp_52 = torch.stack([tmp_50, tmp_51], -1)
        tmp_50 = tmp_51 = None
        tmp_53 = tmp_52.reshape((1, 3, 256, 64))
        tmp_52 = None
        tmp_54 = tmp_53 * tmp_46
        tmp_53 = tmp_46 = None
        tmp_55 = tmp_48 + tmp_54
        tmp_48 = tmp_54 = None
        tmp_56 = torch.cat([tmp_43, tmp_55], dim=2)
        tmp_43 = tmp_55 = None
        tmp_57 = tmp_56.type_as(tmp_27)
        tmp_56 = None
        tmp_58 = torch.nn.functional.scaled_dot_product_attention(tmp_42, tmp_57, tmp_27, attn_mask=None, dropout_p=0.0)
        tmp_42 = tmp_57 = tmp_27 = None
        tmp_59 = tmp_58.transpose(1, 2)
        tmp_58 = None
        tmp_60 = tmp_59.reshape(1, 257, 192)
        tmp_59 = None
        return (tmp_18, tmp_60)
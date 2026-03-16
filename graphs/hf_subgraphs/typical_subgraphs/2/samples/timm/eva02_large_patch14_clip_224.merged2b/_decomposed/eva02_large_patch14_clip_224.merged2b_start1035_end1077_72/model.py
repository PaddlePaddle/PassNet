import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6.reshape(1, 257, 16, -1)
        tmp_7 = tmp_6.transpose(1, 2)
        tmp_6 = None
        tmp_8 = torch.nn.functional.linear(in_7, tmp_0, None)
        tmp_0 = None
        tmp_9 = tmp_8.reshape(1, 257, 16, -1)
        tmp_8 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = torch.nn.functional.linear(in_7, tmp_4, tmp_3)
        tmp_4 = tmp_3 = None
        tmp_12 = tmp_11.reshape(1, 257, 16, -1)
        tmp_11 = None
        tmp_13 = tmp_12.transpose(1, 2)
        tmp_12 = None
        tmp_14 = tmp_7[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
        tmp_15 = tmp_7[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_7 = None
        tmp_16 = tmp_5.tensor_split(2, -1)
        tmp_17 = tmp_16[0]
        tmp_18 = tmp_16[1]
        tmp_16 = None
        tmp_19 = tmp_15 * tmp_18
        tmp_18 = None
        tmp_20 = tmp_15[Ellipsis, slice(1, None, 2)]
        tmp_21 = -tmp_20
        tmp_20 = None
        tmp_22 = tmp_15[Ellipsis, slice(None, None, 2)]
        tmp_15 = None
        tmp_23 = torch.stack([tmp_21, tmp_22], -1)
        tmp_21 = tmp_22 = None
        tmp_24 = tmp_23.reshape((1, 16, 256, 64))
        tmp_23 = None
        tmp_25 = tmp_24 * tmp_17
        tmp_24 = tmp_17 = None
        tmp_26 = tmp_19 + tmp_25
        tmp_19 = tmp_25 = None
        tmp_27 = torch.cat([tmp_14, tmp_26], dim=2)
        tmp_14 = tmp_26 = None
        tmp_28 = tmp_27.type_as(tmp_13)
        tmp_27 = None
        tmp_29 = tmp_10[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
        tmp_30 = tmp_10[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_10 = None
        tmp_31 = tmp_5.tensor_split(2, -1)
        tmp_5 = None
        tmp_32 = tmp_31[0]
        tmp_33 = tmp_31[1]
        tmp_31 = None
        tmp_34 = tmp_30 * tmp_33
        tmp_33 = None
        tmp_35 = tmp_30[Ellipsis, slice(1, None, 2)]
        tmp_36 = -tmp_35
        tmp_35 = None
        tmp_37 = tmp_30[Ellipsis, slice(None, None, 2)]
        tmp_30 = None
        tmp_38 = torch.stack([tmp_36, tmp_37], -1)
        tmp_36 = tmp_37 = None
        tmp_39 = tmp_38.reshape((1, 16, 256, 64))
        tmp_38 = None
        tmp_40 = tmp_39 * tmp_32
        tmp_39 = tmp_32 = None
        tmp_41 = tmp_34 + tmp_40
        tmp_34 = tmp_40 = None
        tmp_42 = torch.cat([tmp_29, tmp_41], dim=2)
        tmp_29 = tmp_41 = None
        tmp_43 = tmp_42.type_as(tmp_13)
        tmp_42 = None
        tmp_44 = torch.nn.functional.scaled_dot_product_attention(tmp_28, tmp_43, tmp_13, attn_mask=None, dropout_p=0.0)
        tmp_28 = tmp_43 = tmp_13 = None
        tmp_45 = tmp_44.transpose(1, 2)
        tmp_44 = None
        tmp_46 = tmp_45.reshape(1, 257, 1024)
        tmp_45 = None
        tmp_47 = torch.nn.functional.layer_norm(tmp_46, (1024,), tmp_2, tmp_1, 1e-06)
        tmp_46 = tmp_2 = tmp_1 = None
        return (tmp_47,)
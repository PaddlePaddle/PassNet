import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4.reshape(1, 257, 12, -1)
        tmp_5 = tmp_4.transpose(1, 2)
        tmp_4 = None
        tmp_6 = torch.nn.functional.linear(in_5, tmp_0, None)
        tmp_0 = None
        tmp_7 = tmp_6.reshape(1, 257, 12, -1)
        tmp_6 = None
        tmp_8 = tmp_7.transpose(1, 2)
        tmp_7 = None
        tmp_9 = torch.nn.functional.linear(in_5, tmp_2, tmp_1)
        tmp_2 = tmp_1 = None
        tmp_10 = tmp_9.reshape(1, 257, 12, -1)
        tmp_9 = None
        tmp_11 = tmp_10.transpose(1, 2)
        tmp_10 = None
        tmp_12 = tmp_5[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
        tmp_13 = tmp_5[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_5 = None
        tmp_14 = tmp_3.tensor_split(2, -1)
        tmp_15 = tmp_14[0]
        tmp_16 = tmp_14[1]
        tmp_14 = None
        tmp_17 = tmp_13 * tmp_16
        tmp_16 = None
        tmp_18 = tmp_13[Ellipsis, slice(1, None, 2)]
        tmp_19 = -tmp_18
        tmp_18 = None
        tmp_20 = tmp_13[Ellipsis, slice(None, None, 2)]
        tmp_13 = None
        tmp_21 = torch.stack([tmp_19, tmp_20], -1)
        tmp_19 = tmp_20 = None
        tmp_22 = tmp_21.reshape((1, 12, 256, 64))
        tmp_21 = None
        tmp_23 = tmp_22 * tmp_15
        tmp_22 = tmp_15 = None
        tmp_24 = tmp_17 + tmp_23
        tmp_17 = tmp_23 = None
        tmp_25 = torch.cat([tmp_12, tmp_24], dim=2)
        tmp_12 = tmp_24 = None
        tmp_26 = tmp_25.type_as(tmp_11)
        tmp_25 = None
        tmp_27 = tmp_8[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
        tmp_28 = tmp_8[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_8 = None
        tmp_29 = tmp_3.tensor_split(2, -1)
        tmp_3 = None
        tmp_30 = tmp_29[0]
        tmp_31 = tmp_29[1]
        tmp_29 = None
        tmp_32 = tmp_28 * tmp_31
        tmp_31 = None
        tmp_33 = tmp_28[Ellipsis, slice(1, None, 2)]
        tmp_34 = -tmp_33
        tmp_33 = None
        tmp_35 = tmp_28[Ellipsis, slice(None, None, 2)]
        tmp_28 = None
        tmp_36 = torch.stack([tmp_34, tmp_35], -1)
        tmp_34 = tmp_35 = None
        tmp_37 = tmp_36.reshape((1, 12, 256, 64))
        tmp_36 = None
        tmp_38 = tmp_37 * tmp_30
        tmp_37 = tmp_30 = None
        tmp_39 = tmp_32 + tmp_38
        tmp_32 = tmp_38 = None
        tmp_40 = torch.cat([tmp_27, tmp_39], dim=2)
        tmp_27 = tmp_39 = None
        tmp_41 = tmp_40.type_as(tmp_11)
        tmp_40 = None
        tmp_42 = torch.nn.functional.scaled_dot_product_attention(tmp_26, tmp_41, tmp_11, attn_mask=None, dropout_p=0.0)
        tmp_26 = tmp_41 = tmp_11 = None
        tmp_43 = tmp_42.transpose(1, 2)
        tmp_42 = None
        tmp_44 = tmp_43.reshape(1, 257, 768)
        tmp_43 = None
        return (tmp_44,)
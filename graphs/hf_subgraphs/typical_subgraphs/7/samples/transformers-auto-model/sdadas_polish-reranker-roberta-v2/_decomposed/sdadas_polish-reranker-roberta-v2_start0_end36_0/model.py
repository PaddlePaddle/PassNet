import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13):
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
        tmp_13 = in_13
        tmp_14 = tmp_2[slice(None, None, None), slice(None, 128, None)]
        tmp_2 = None
        tmp_15 = tmp_14.expand(1, 128)
        tmp_14 = None
        tmp_16 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_17 = tmp_16.to(dtype=torch.bfloat16)
        tmp_16 = None
        tmp_18 = 1.0 - tmp_17
        tmp_17 = None
        tmp_19 = tmp_18 * -3.3895313892515355e+38
        tmp_18 = None
        tmp_20 = tmp_1.ne(1)
        tmp_21 = tmp_20.int()
        tmp_20 = None
        tmp_22 = torch.cumsum(tmp_21, dim=1)
        tmp_23 = tmp_22.type_as(tmp_21)
        tmp_22 = None
        tmp_24 = tmp_23 + 0
        tmp_23 = None
        tmp_25 = tmp_24 * tmp_21
        tmp_24 = tmp_21 = None
        tmp_26 = tmp_25.long()
        tmp_25 = None
        tmp_27 = tmp_26 + 1
        tmp_26 = None
        tmp_28 = torch.nn.functional.embedding(tmp_1, tmp_7, 1, None, 2.0, False, False)
        tmp_1 = tmp_7 = None
        tmp_29 = torch.nn.functional.embedding(tmp_15, tmp_6, None, None, 2.0, False, False)
        tmp_15 = tmp_6 = None
        tmp_30 = tmp_28 + tmp_29
        tmp_28 = tmp_29 = None
        tmp_31 = torch.nn.functional.embedding(tmp_27, tmp_5, 1, None, 2.0, False, False)
        tmp_27 = tmp_5 = None
        tmp_30 += tmp_31
        tmp_32 = tmp_30
        tmp_30 = tmp_31 = None
        tmp_33 = torch.nn.functional.layer_norm(tmp_32, (1024,), tmp_4, tmp_3, 1e-05)
        tmp_32 = tmp_4 = tmp_3 = None
        tmp_34 = torch.nn.functional.dropout(tmp_33, 0.1, False, False)
        tmp_33 = None
        tmp_35 = torch.nn.functional.linear(tmp_34, tmp_11, tmp_10)
        tmp_11 = tmp_10 = None
        tmp_36 = torch.nn.functional.linear(tmp_34, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_37 = tmp_36.view((64, 128, 16, 64))
        tmp_36 = None
        tmp_38 = tmp_37.permute(0, 2, 1, 3)
        tmp_37 = None
        tmp_39 = torch.nn.functional.linear(tmp_34, tmp_13, tmp_12)
        tmp_13 = tmp_12 = None
        tmp_40 = tmp_39.view((64, 128, 16, 64))
        tmp_39 = None
        tmp_41 = tmp_40.permute(0, 2, 1, 3)
        tmp_40 = None
        tmp_42 = tmp_35.view((64, 128, 16, 64))
        tmp_35 = None
        tmp_43 = tmp_42.permute(0, 2, 1, 3)
        tmp_42 = None
        tmp_44 = tmp_43.contiguous()
        tmp_43 = None
        tmp_45 = tmp_38.contiguous()
        tmp_38 = None
        tmp_46 = tmp_41.contiguous()
        tmp_41 = None
        tmp_47 = torch.nn.functional.scaled_dot_product_attention(tmp_44, tmp_45, tmp_46, attn_mask=tmp_19, dropout_p=0.0, is_causal=False)
        tmp_44 = tmp_45 = tmp_46 = None
        tmp_48 = tmp_47.transpose(1, 2)
        tmp_47 = None
        tmp_49 = tmp_48.reshape(64, 128, 1024)
        tmp_48 = None
        return (tmp_34, tmp_19, tmp_49)
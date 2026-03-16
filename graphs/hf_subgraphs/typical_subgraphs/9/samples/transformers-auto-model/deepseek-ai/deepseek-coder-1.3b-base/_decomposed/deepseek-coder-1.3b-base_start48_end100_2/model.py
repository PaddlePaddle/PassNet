import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = w_6
        tmp_8 = torch.nn.functional.linear(in_3, tmp_6, None)
        tmp_6 = None
        tmp_9 = tmp_8.view((1, 3, -1, 128))
        tmp_8 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = in_2.unsqueeze(1)
        tmp_12 = in_6.unsqueeze(1)
        tmp_13 = in_5 * tmp_11
        tmp_14 = in_5[Ellipsis, slice(None, 64, None)]
        tmp_15 = in_5[Ellipsis, slice(64, None, None)]
        tmp_16 = -tmp_15
        tmp_15 = None
        tmp_17 = torch.cat((tmp_16, tmp_14), dim=-1)
        tmp_16 = tmp_14 = None
        tmp_18 = tmp_17 * tmp_12
        tmp_17 = None
        tmp_19 = tmp_13 + tmp_18
        tmp_13 = tmp_18 = None
        tmp_20 = in_4 * tmp_11
        tmp_11 = None
        tmp_21 = in_4[Ellipsis, slice(None, 64, None)]
        tmp_22 = in_4[Ellipsis, slice(64, None, None)]
        tmp_23 = -tmp_22
        tmp_22 = None
        tmp_24 = torch.cat((tmp_23, tmp_21), dim=-1)
        tmp_23 = tmp_21 = None
        tmp_25 = tmp_24 * tmp_12
        tmp_24 = tmp_12 = None
        tmp_26 = tmp_20 + tmp_25
        tmp_20 = tmp_25 = None
        tmp_27 = in_1[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 3, None)]
        tmp_28 = tmp_19.contiguous()
        tmp_19 = None
        tmp_29 = tmp_26.contiguous()
        tmp_30 = tmp_10.contiguous()
        tmp_31 = torch.nn.functional.scaled_dot_product_attention(tmp_28, tmp_29, tmp_30, attn_mask=tmp_27, dropout_p=0.0, scale=0.08838834764831845, is_causal=False)
        tmp_28 = tmp_29 = tmp_30 = tmp_27 = None
        tmp_32 = tmp_31.transpose(1, 2)
        tmp_31 = None
        tmp_33 = tmp_32.contiguous()
        tmp_32 = None
        tmp_34 = tmp_33.reshape(1, 3, -1)
        tmp_33 = None
        tmp_35 = tmp_34.contiguous()
        tmp_34 = None
        tmp_36 = torch.nn.functional.linear(tmp_35, tmp_5, None)
        tmp_35 = tmp_5 = None
        tmp_37 = tmp_0 + tmp_36
        tmp_0 = tmp_36 = None
        tmp_38 = tmp_37.to(torch.float32)
        tmp_39 = tmp_38.pow(2)
        tmp_40 = tmp_39.mean(-1, keepdim=True)
        tmp_39 = None
        tmp_41 = tmp_40 + 1e-06
        tmp_40 = None
        tmp_42 = torch.rsqrt(tmp_41)
        tmp_41 = None
        tmp_43 = tmp_38 * tmp_42
        tmp_38 = tmp_42 = None
        tmp_44 = tmp_43.to(torch.bfloat16)
        tmp_43 = None
        tmp_45 = tmp_4 * tmp_44
        tmp_4 = tmp_44 = None
        tmp_46 = torch.nn.functional.linear(tmp_45, tmp_2, None)
        tmp_2 = None
        tmp_47 = torch.nn.functional.silu(tmp_46, inplace=False)
        tmp_46 = None
        tmp_48 = torch.nn.functional.linear(tmp_45, tmp_3, None)
        tmp_45 = tmp_3 = None
        tmp_49 = tmp_47 * tmp_48
        tmp_47 = tmp_48 = None
        tmp_50 = torch.nn.functional.linear(tmp_49, tmp_1, None)
        tmp_49 = tmp_1 = None
        tmp_51 = tmp_37 + tmp_50
        tmp_37 = tmp_50 = None
        tmp_52 = tmp_51.to(torch.float32)
        tmp_53 = tmp_52.pow(2)
        tmp_54 = tmp_53.mean(-1, keepdim=True)
        tmp_53 = None
        tmp_55 = tmp_54 + 1e-06
        tmp_54 = None
        tmp_56 = torch.rsqrt(tmp_55)
        tmp_55 = None
        tmp_57 = tmp_52 * tmp_56
        tmp_52 = tmp_56 = None
        tmp_58 = tmp_57.to(torch.bfloat16)
        tmp_57 = None
        tmp_59 = tmp_7 * tmp_58
        tmp_7 = tmp_58 = None
        return (tmp_59, tmp_51, tmp_26, tmp_10)
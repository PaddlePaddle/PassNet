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
        tmp_7 = torch.nn.functional.linear(in_10, tmp_5, None)
        tmp_5 = None
        tmp_8 = tmp_7.view((64, 128, -1, 64))
        tmp_7 = None
        tmp_9 = tmp_8.transpose(1, 2)
        tmp_8 = None
        tmp_10 = in_8.unsqueeze(1)
        tmp_11 = in_13.unsqueeze(1)
        tmp_12 = in_12 * tmp_10
        tmp_13 = in_12[Ellipsis, slice(None, 32, None)]
        tmp_14 = in_12[Ellipsis, slice(32, None, None)]
        tmp_15 = -tmp_14
        tmp_14 = None
        tmp_16 = torch.cat((tmp_15, tmp_13), dim=-1)
        tmp_15 = tmp_13 = None
        tmp_17 = tmp_16 * tmp_11
        tmp_16 = None
        tmp_18 = tmp_12 + tmp_17
        tmp_12 = tmp_17 = None
        tmp_19 = in_11 * tmp_10
        tmp_10 = None
        tmp_20 = in_11[Ellipsis, slice(None, 32, None)]
        tmp_21 = in_11[Ellipsis, slice(32, None, None)]
        tmp_22 = -tmp_21
        tmp_21 = None
        tmp_23 = torch.cat((tmp_22, tmp_20), dim=-1)
        tmp_22 = tmp_20 = None
        tmp_24 = tmp_23 * tmp_11
        tmp_23 = tmp_11 = None
        tmp_25 = tmp_19 + tmp_24
        tmp_19 = tmp_24 = None
        tmp_26 = tmp_25[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
        tmp_25 = None
        tmp_27 = tmp_26.expand(64, 4, 8, 128, 64)
        tmp_26 = None
        tmp_28 = tmp_27.reshape(64, 32, 128, 64)
        tmp_27 = None
        tmp_29 = tmp_9[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
        tmp_9 = None
        tmp_30 = tmp_29.expand(64, 4, 8, 128, 64)
        tmp_29 = None
        tmp_31 = tmp_30.reshape(64, 32, 128, 64)
        tmp_30 = None
        tmp_32 = in_7[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None)]
        tmp_33 = tmp_18.contiguous()
        tmp_18 = None
        tmp_34 = tmp_28.contiguous()
        tmp_28 = None
        tmp_35 = tmp_31.contiguous()
        tmp_31 = None
        tmp_36 = torch.nn.functional.scaled_dot_product_attention(tmp_33, tmp_34, tmp_35, attn_mask=tmp_32, dropout_p=0.0, scale=0.125, is_causal=False)
        tmp_33 = tmp_34 = tmp_35 = tmp_32 = None
        tmp_37 = tmp_36.transpose(1, 2)
        tmp_36 = None
        tmp_38 = tmp_37.contiguous()
        tmp_37 = None
        tmp_39 = tmp_38.reshape(64, 128, -1)
        tmp_38 = None
        tmp_40 = tmp_39.contiguous()
        tmp_39 = None
        tmp_41 = torch.nn.functional.linear(tmp_40, tmp_4, None)
        tmp_40 = tmp_4 = None
        tmp_42 = in_9 + tmp_41
        tmp_41 = None
        tmp_43 = tmp_42.to(torch.float32)
        tmp_44 = tmp_43.pow(2)
        tmp_45 = tmp_44.mean(-1, keepdim=True)
        tmp_44 = None
        tmp_46 = tmp_45 + 1e-05
        tmp_45 = None
        tmp_47 = torch.rsqrt(tmp_46)
        tmp_46 = None
        tmp_48 = tmp_43 * tmp_47
        tmp_43 = tmp_47 = None
        tmp_49 = tmp_48.to(torch.float32)
        tmp_48 = None
        tmp_50 = tmp_3 * tmp_49
        tmp_3 = tmp_49 = None
        tmp_51 = torch.nn.functional.linear(tmp_50, tmp_1, None)
        tmp_1 = None
        tmp_52 = torch.nn.functional.silu(tmp_51, inplace=False)
        tmp_51 = None
        tmp_53 = torch.nn.functional.linear(tmp_50, tmp_2, None)
        tmp_50 = tmp_2 = None
        tmp_54 = tmp_52 * tmp_53
        tmp_52 = tmp_53 = None
        tmp_55 = torch.nn.functional.linear(tmp_54, tmp_0, None)
        tmp_54 = tmp_0 = None
        tmp_56 = tmp_42 + tmp_55
        tmp_42 = tmp_55 = None
        tmp_57 = tmp_56.to(torch.float32)
        tmp_58 = tmp_57.pow(2)
        tmp_59 = tmp_58.mean(-1, keepdim=True)
        tmp_58 = None
        tmp_60 = tmp_59 + 1e-05
        tmp_59 = None
        tmp_61 = torch.rsqrt(tmp_60)
        tmp_60 = None
        tmp_62 = tmp_57 * tmp_61
        tmp_57 = tmp_61 = None
        tmp_63 = tmp_62.to(torch.float32)
        tmp_62 = None
        tmp_64 = tmp_6 * tmp_63
        tmp_6 = tmp_63 = None
        return (tmp_56, tmp_64)
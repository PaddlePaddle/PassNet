import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = torch.nn.functional.linear(in_2, tmp_6, None)
        tmp_6 = None
        tmp_9 = tmp_8.view((1, 2, -1, 128))
        tmp_8 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = in_3[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
        tmp_12 = tmp_11.expand(1, 4, 4, 2, 128)
        tmp_11 = None
        tmp_13 = tmp_12.reshape(1, 16, 2, 128)
        tmp_12 = None
        tmp_14 = tmp_10[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
        tmp_10 = None
        tmp_15 = tmp_14.expand(1, 4, 4, 2, 128)
        tmp_14 = None
        tmp_16 = tmp_15.reshape(1, 16, 2, 128)
        tmp_15 = None
        tmp_17 = in_0[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 2, None)]
        tmp_18 = in_4.contiguous()
        tmp_19 = tmp_13.contiguous()
        tmp_13 = None
        tmp_20 = tmp_16.contiguous()
        tmp_16 = None
        tmp_21 = torch.nn.functional.scaled_dot_product_attention(tmp_18, tmp_19, tmp_20, attn_mask=tmp_17, dropout_p=0.0, scale=0.08838834764831845, is_causal=False)
        tmp_18 = tmp_19 = tmp_20 = tmp_17 = None
        tmp_22 = tmp_21.transpose(1, 2)
        tmp_21 = None
        tmp_23 = tmp_22.contiguous()
        tmp_22 = None
        tmp_24 = tmp_23.reshape(1, 2, -1)
        tmp_23 = None
        tmp_25 = tmp_24.contiguous()
        tmp_24 = None
        tmp_26 = torch.nn.functional.linear(tmp_25, tmp_5, None)
        tmp_25 = tmp_5 = None
        tmp_27 = in_1 + tmp_26
        tmp_26 = None
        tmp_28 = tmp_27.to(torch.float32)
        tmp_29 = tmp_28.pow(2)
        tmp_30 = tmp_29.mean(-1, keepdim=True)
        tmp_29 = None
        tmp_31 = tmp_30 + 1e-06
        tmp_30 = None
        tmp_32 = torch.rsqrt(tmp_31)
        tmp_31 = None
        tmp_33 = tmp_28 * tmp_32
        tmp_28 = tmp_32 = None
        tmp_34 = tmp_33.to(torch.bfloat16)
        tmp_33 = None
        tmp_35 = tmp_4 * tmp_34
        tmp_4 = tmp_34 = None
        tmp_36 = torch.nn.functional.linear(tmp_35, tmp_2, None)
        tmp_2 = None
        tmp_37 = torch.nn.functional.silu(tmp_36, inplace=False)
        tmp_36 = None
        tmp_38 = torch.nn.functional.linear(tmp_35, tmp_3, None)
        tmp_35 = tmp_3 = None
        tmp_39 = tmp_37 * tmp_38
        tmp_37 = tmp_38 = None
        tmp_40 = torch.nn.functional.linear(tmp_39, tmp_1, None)
        tmp_39 = tmp_1 = None
        tmp_41 = tmp_27 + tmp_40
        tmp_27 = tmp_40 = None
        tmp_42 = tmp_41.to(torch.float32)
        tmp_41 = None
        tmp_43 = tmp_42.pow(2)
        tmp_44 = tmp_43.mean(-1, keepdim=True)
        tmp_43 = None
        tmp_45 = tmp_44 + 1e-06
        tmp_44 = None
        tmp_46 = torch.rsqrt(tmp_45)
        tmp_45 = None
        tmp_47 = tmp_42 * tmp_46
        tmp_42 = tmp_46 = None
        tmp_48 = tmp_47.to(torch.bfloat16)
        tmp_47 = None
        tmp_49 = tmp_7 * tmp_48
        tmp_7 = tmp_48 = None
        tmp_50 = tmp_49[slice(None, None, None), slice(0, None, None), slice(None, None, None)]
        tmp_49 = None
        tmp_51 = torch.nn.functional.linear(tmp_50, tmp_0, None)
        tmp_50 = tmp_0 = None
        return (tmp_51,)
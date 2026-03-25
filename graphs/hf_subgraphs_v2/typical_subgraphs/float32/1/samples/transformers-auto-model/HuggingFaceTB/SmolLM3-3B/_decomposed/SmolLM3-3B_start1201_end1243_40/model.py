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
        tmp_7 = torch.nn.functional.linear(in_9, tmp_5, None)
        tmp_5 = None
        tmp_8 = tmp_7.view((1, 512, -1, 128))
        tmp_7 = None
        tmp_9 = tmp_8.transpose(1, 2)
        tmp_8 = None
        tmp_10 = in_10[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
        tmp_11 = tmp_10.expand(1, 4, 4, 512, 128)
        tmp_10 = None
        tmp_12 = tmp_11.reshape(1, 16, 512, 128)
        tmp_11 = None
        tmp_13 = tmp_9[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
        tmp_9 = None
        tmp_14 = tmp_13.expand(1, 4, 4, 512, 128)
        tmp_13 = None
        tmp_15 = tmp_14.reshape(1, 16, 512, 128)
        tmp_14 = None
        tmp_16 = in_7[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 512, None)]
        tmp_17 = in_11.contiguous()
        tmp_18 = tmp_12.contiguous()
        tmp_12 = None
        tmp_19 = tmp_15.contiguous()
        tmp_15 = None
        tmp_20 = torch.nn.functional.scaled_dot_product_attention(tmp_17, tmp_18, tmp_19, attn_mask=tmp_16, dropout_p=0.0, scale=0.08838834764831845, is_causal=False)
        tmp_17 = tmp_18 = tmp_19 = tmp_16 = None
        tmp_21 = tmp_20.transpose(1, 2)
        tmp_20 = None
        tmp_22 = tmp_21.contiguous()
        tmp_21 = None
        tmp_23 = tmp_22.reshape(1, 512, -1)
        tmp_22 = None
        tmp_24 = tmp_23.contiguous()
        tmp_23 = None
        tmp_25 = torch.nn.functional.linear(tmp_24, tmp_4, None)
        tmp_24 = tmp_4 = None
        tmp_26 = in_8 + tmp_25
        tmp_25 = None
        tmp_27 = tmp_26.to(torch.float32)
        tmp_28 = tmp_27.pow(2)
        tmp_29 = tmp_28.mean(-1, keepdim=True)
        tmp_28 = None
        tmp_30 = tmp_29 + 1e-06
        tmp_29 = None
        tmp_31 = torch.rsqrt(tmp_30)
        tmp_30 = None
        tmp_32 = tmp_27 * tmp_31
        tmp_27 = tmp_31 = None
        tmp_33 = tmp_32.to(torch.bfloat16)
        tmp_32 = None
        tmp_34 = tmp_3 * tmp_33
        tmp_3 = tmp_33 = None
        tmp_35 = torch.nn.functional.linear(tmp_34, tmp_1, None)
        tmp_1 = None
        tmp_36 = torch.nn.functional.silu(tmp_35, inplace=False)
        tmp_35 = None
        tmp_37 = torch.nn.functional.linear(tmp_34, tmp_2, None)
        tmp_34 = tmp_2 = None
        tmp_38 = tmp_36 * tmp_37
        tmp_36 = tmp_37 = None
        tmp_39 = torch.nn.functional.linear(tmp_38, tmp_0, None)
        tmp_38 = tmp_0 = None
        tmp_40 = tmp_26 + tmp_39
        tmp_26 = tmp_39 = None
        tmp_41 = tmp_40.to(torch.float32)
        tmp_42 = tmp_41.pow(2)
        tmp_43 = tmp_42.mean(-1, keepdim=True)
        tmp_42 = None
        tmp_44 = tmp_43 + 1e-06
        tmp_43 = None
        tmp_45 = torch.rsqrt(tmp_44)
        tmp_44 = None
        tmp_46 = tmp_41 * tmp_45
        tmp_41 = tmp_45 = None
        tmp_47 = tmp_46.to(torch.bfloat16)
        tmp_46 = None
        tmp_48 = tmp_6 * tmp_47
        tmp_6 = tmp_47 = None
        return (tmp_40, tmp_48)
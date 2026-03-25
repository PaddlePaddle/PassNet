import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0, w_1, w_2, w_3, w_4, w_5, w_6, in_1, in_2, in_3, in_4, in_5, in_6):
        linear = torch.nn.functional.linear(in_3, w_5, None);  in_3 = w_5 = None
        tmp_9 = linear.view((1, 3, -1, 128));  linear = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        tmp_11 = in_2.unsqueeze(1);  in_2 = None
        tmp_12 = in_6.unsqueeze(1);  in_6 = None
        tmp_13 = in_5 * tmp_11
        tmp_14 = in_5[(Ellipsis, slice(None, 64, None))]
        tmp_15 = in_5[(Ellipsis, slice(64, None, None))];  in_5 = None
        tmp_16 = -tmp_15;  tmp_15 = None
        tmp_17 = torch.cat((tmp_16, tmp_14), dim = -1);  tmp_16 = tmp_14 = None
        tmp_18 = tmp_17 * tmp_12;  tmp_17 = None
        tmp_19 = tmp_13 + tmp_18;  tmp_13 = tmp_18 = None
        tmp_20 = in_4 * tmp_11;  tmp_11 = None
        tmp_21 = in_4[(Ellipsis, slice(None, 64, None))]
        tmp_22 = in_4[(Ellipsis, slice(64, None, None))];  in_4 = None
        tmp_23 = -tmp_22;  tmp_22 = None
        tmp_24 = torch.cat((tmp_23, tmp_21), dim = -1);  tmp_23 = tmp_21 = None
        tmp_25 = tmp_24 * tmp_12;  tmp_24 = tmp_12 = None
        tmp_26 = tmp_20 + tmp_25;  tmp_20 = tmp_25 = None
        tmp_27 = in_1[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 3, None))];  in_1 = None
        tmp_28 = tmp_19.contiguous();  tmp_19 = None
        tmp_29 = tmp_26.contiguous()
        tmp_30 = tmp_10.contiguous()
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_28, tmp_29, tmp_30, attn_mask = tmp_27, dropout_p = 0.0, scale = 0.08838834764831845, is_causal = False);  tmp_28 = tmp_29 = tmp_30 = tmp_27 = None
        tmp_32 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_33 = tmp_32.contiguous();  tmp_32 = None
        tmp_34 = tmp_33.reshape(1, 3, -1);  tmp_33 = None
        tmp_35 = tmp_34.contiguous();  tmp_34 = None
        linear_1 = torch.nn.functional.linear(tmp_35, w_4, None);  tmp_35 = w_4 = None
        tmp_37 = in_0 + linear_1;  in_0 = linear_1 = None
        tmp_38 = tmp_37.to(torch.float32)
        tmp_39 = tmp_38.pow(2)
        tmp_40 = tmp_39.mean(-1, keepdim = True);  tmp_39 = None
        tmp_41 = tmp_40 + 1e-06;  tmp_40 = None
        tmp_42 = torch.rsqrt(tmp_41);  tmp_41 = None
        tmp_43 = tmp_38 * tmp_42;  tmp_38 = tmp_42 = None
        tmp_44 = tmp_43.to(torch.bfloat16);  tmp_43 = None
        tmp_45 = w_3 * tmp_44;  w_3 = tmp_44 = None
        linear_2 = torch.nn.functional.linear(tmp_45, w_1, None);  w_1 = None
        tmp_47 = torch.nn.functional.silu(linear_2, inplace = False);  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_45, w_2, None);  tmp_45 = w_2 = None
        tmp_49 = tmp_47 * linear_3;  tmp_47 = linear_3 = None
        linear_4 = torch.nn.functional.linear(tmp_49, w_0, None);  tmp_49 = w_0 = None
        tmp_51 = tmp_37 + linear_4;  tmp_37 = linear_4 = None
        tmp_52 = tmp_51.to(torch.float32)
        tmp_53 = tmp_52.pow(2)
        tmp_54 = tmp_53.mean(-1, keepdim = True);  tmp_53 = None
        tmp_55 = tmp_54 + 1e-06;  tmp_54 = None
        tmp_56 = torch.rsqrt(tmp_55);  tmp_55 = None
        tmp_57 = tmp_52 * tmp_56;  tmp_52 = tmp_56 = None
        tmp_58 = tmp_57.to(torch.bfloat16);  tmp_57 = None
        tmp_59 = w_6 * tmp_58;  w_6 = tmp_58 = None
        return (tmp_59, tmp_51, tmp_26, tmp_10)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12):
        linear = torch.nn.functional.linear(in_10, in_6, None);  in_10 = in_6 = None
        tmp_9 = linear.view((1, 64, -1, 128));  linear = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        tmp_11 = in_11[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None))];  in_11 = None
        tmp_12 = tmp_11.expand(1, 4, 4, 64, 128);  tmp_11 = None
        tmp_13 = tmp_12.reshape(1, 16, 64, 128);  tmp_12 = None
        tmp_14 = tmp_10[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None))];  tmp_10 = None
        tmp_15 = tmp_14.expand(1, 4, 4, 64, 128);  tmp_14 = None
        tmp_16 = tmp_15.reshape(1, 16, 64, 128);  tmp_15 = None
        tmp_17 = in_8[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))];  in_8 = None
        tmp_18 = in_12.contiguous();  in_12 = None
        tmp_19 = tmp_13.contiguous();  tmp_13 = None
        tmp_20 = tmp_16.contiguous();  tmp_16 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_18, tmp_19, tmp_20, attn_mask = tmp_17, dropout_p = 0.0, scale = 0.08838834764831845, is_causal = False);  tmp_18 = tmp_19 = tmp_20 = tmp_17 = None
        tmp_22 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_23 = tmp_22.contiguous();  tmp_22 = None
        tmp_24 = tmp_23.reshape(1, 64, -1);  tmp_23 = None
        tmp_25 = tmp_24.contiguous();  tmp_24 = None
        linear_1 = torch.nn.functional.linear(tmp_25, in_5, None);  tmp_25 = in_5 = None
        tmp_27 = in_9 + linear_1;  in_9 = linear_1 = None
        tmp_28 = tmp_27.to(torch.float32)
        tmp_29 = tmp_28.pow(2)
        tmp_30 = tmp_29.mean(-1, keepdim = True);  tmp_29 = None
        tmp_31 = tmp_30 + 1e-06;  tmp_30 = None
        tmp_32 = torch.rsqrt(tmp_31);  tmp_31 = None
        tmp_33 = tmp_28 * tmp_32;  tmp_28 = tmp_32 = None
        tmp_34 = tmp_33.to(torch.bfloat16);  tmp_33 = None
        tmp_35 = in_4 * tmp_34;  in_4 = tmp_34 = None
        linear_2 = torch.nn.functional.linear(tmp_35, in_2, None);  in_2 = None
        tmp_37 = torch.nn.functional.silu(linear_2, inplace = False);  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_35, in_3, None);  tmp_35 = in_3 = None
        tmp_39 = tmp_37 * linear_3;  tmp_37 = linear_3 = None
        linear_4 = torch.nn.functional.linear(tmp_39, in_1, None);  tmp_39 = in_1 = None
        tmp_41 = tmp_27 + linear_4;  tmp_27 = linear_4 = None
        tmp_42 = tmp_41.to(torch.float32);  tmp_41 = None
        tmp_43 = tmp_42.pow(2)
        tmp_44 = tmp_43.mean(-1, keepdim = True);  tmp_43 = None
        tmp_45 = tmp_44 + 1e-06;  tmp_44 = None
        tmp_46 = torch.rsqrt(tmp_45);  tmp_45 = None
        tmp_47 = tmp_42 * tmp_46;  tmp_42 = tmp_46 = None
        tmp_48 = tmp_47.to(torch.bfloat16);  tmp_47 = None
        tmp_49 = in_7 * tmp_48;  in_7 = tmp_48 = None
        tmp_50 = tmp_49[(slice(None, None, None), slice(0, None, None), slice(None, None, None))];  tmp_49 = None
        linear_5 = torch.nn.functional.linear(tmp_50, in_0, None);  tmp_50 = in_0 = None
        return (linear_5,)
        
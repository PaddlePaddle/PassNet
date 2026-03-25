import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13):
        linear = torch.nn.functional.linear(in_10, in_6, None);  in_10 = in_6 = None
        tmp_8 = linear.view((32, 64, -1, 64));  linear = None
        tmp_9 = tmp_8.transpose(1, 2);  tmp_8 = None
        tmp_10 = in_8.unsqueeze(1);  in_8 = None
        tmp_11 = in_13.unsqueeze(1);  in_13 = None
        tmp_12 = in_12 * tmp_10
        tmp_13 = in_12[(Ellipsis, slice(None, 32, None))]
        tmp_14 = in_12[(Ellipsis, slice(32, None, None))];  in_12 = None
        tmp_15 = -tmp_14;  tmp_14 = None
        tmp_16 = torch.cat((tmp_15, tmp_13), dim = -1);  tmp_15 = tmp_13 = None
        tmp_17 = tmp_16 * tmp_11;  tmp_16 = None
        tmp_18 = tmp_12 + tmp_17;  tmp_12 = tmp_17 = None
        tmp_19 = in_11 * tmp_10;  tmp_10 = None
        tmp_20 = in_11[(Ellipsis, slice(None, 32, None))]
        tmp_21 = in_11[(Ellipsis, slice(32, None, None))];  in_11 = None
        tmp_22 = -tmp_21;  tmp_21 = None
        tmp_23 = torch.cat((tmp_22, tmp_20), dim = -1);  tmp_22 = tmp_20 = None
        tmp_24 = tmp_23 * tmp_11;  tmp_23 = tmp_11 = None
        tmp_25 = tmp_19 + tmp_24;  tmp_19 = tmp_24 = None
        tmp_26 = tmp_25[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None))];  tmp_25 = None
        tmp_27 = tmp_26.expand(32, 4, 8, 64, 64);  tmp_26 = None
        tmp_28 = tmp_27.reshape(32, 32, 64, 64);  tmp_27 = None
        tmp_29 = tmp_9[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None))];  tmp_9 = None
        tmp_30 = tmp_29.expand(32, 4, 8, 64, 64);  tmp_29 = None
        tmp_31 = tmp_30.reshape(32, 32, 64, 64);  tmp_30 = None
        tmp_32 = in_7[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None))];  in_7 = None
        tmp_33 = tmp_18.contiguous();  tmp_18 = None
        tmp_34 = tmp_28.contiguous();  tmp_28 = None
        tmp_35 = tmp_31.contiguous();  tmp_31 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_33, tmp_34, tmp_35, attn_mask = tmp_32, dropout_p = 0.0, scale = 0.125, is_causal = False);  tmp_33 = tmp_34 = tmp_35 = tmp_32 = None
        tmp_37 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_38 = tmp_37.contiguous();  tmp_37 = None
        tmp_39 = tmp_38.reshape(32, 64, -1);  tmp_38 = None
        tmp_40 = tmp_39.contiguous();  tmp_39 = None
        linear_1 = torch.nn.functional.linear(tmp_40, in_5, None);  tmp_40 = in_5 = None
        tmp_42 = in_9 + linear_1;  in_9 = linear_1 = None
        tmp_43 = tmp_42.to(torch.float32)
        tmp_44 = tmp_43.pow(2)
        tmp_45 = tmp_44.mean(-1, keepdim = True);  tmp_44 = None
        tmp_46 = tmp_45 + 1e-05;  tmp_45 = None
        tmp_47 = torch.rsqrt(tmp_46);  tmp_46 = None
        tmp_48 = tmp_43 * tmp_47;  tmp_43 = tmp_47 = None
        tmp_49 = tmp_48.to(torch.float32);  tmp_48 = None
        tmp_50 = in_4 * tmp_49;  in_4 = tmp_49 = None
        to_4 = tmp_50.to(torch.float16)
        linear_2 = torch.nn.functional.linear(to_4, in_2, None);  to_4 = in_2 = None
        tmp_52 = torch.nn.functional.silu(linear_2, inplace = False);  linear_2 = None
        to_5 = tmp_50.to(torch.float16);  tmp_50 = None
        linear_3 = torch.nn.functional.linear(to_5, in_3, None);  to_5 = in_3 = None
        tmp_54 = tmp_52 * linear_3;  tmp_52 = linear_3 = None
        linear_4 = torch.nn.functional.linear(tmp_54, in_1, None);  tmp_54 = in_1 = None
        tmp_56 = tmp_42 + linear_4;  tmp_42 = linear_4 = None
        tmp_57 = tmp_56.to(torch.float32)
        tmp_58 = tmp_57.pow(2)
        tmp_59 = tmp_58.mean(-1, keepdim = True);  tmp_58 = None
        tmp_60 = tmp_59 + 1e-05;  tmp_59 = None
        tmp_61 = torch.rsqrt(tmp_60);  tmp_60 = None
        tmp_62 = tmp_57 * tmp_61;  tmp_57 = tmp_61 = None
        tmp_63 = tmp_62.to(torch.float32);  tmp_62 = None
        tmp_64 = in_0 * tmp_63;  in_0 = tmp_63 = None
        return (tmp_64, tmp_56)
        
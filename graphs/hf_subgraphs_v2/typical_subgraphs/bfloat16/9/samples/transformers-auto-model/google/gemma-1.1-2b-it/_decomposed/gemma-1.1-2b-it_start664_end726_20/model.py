import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        linear = torch.nn.functional.linear(in_3, w_6, None);  in_3 = w_6 = None
        tmp_8 = linear.view((1, 3, -1, 256));  linear = None
        tmp_9 = tmp_8.transpose(1, 2);  tmp_8 = None
        tmp_10 = in_1.unsqueeze(1);  in_1 = None
        tmp_11 = in_6.unsqueeze(1);  in_6 = None
        tmp_12 = in_5 * tmp_10
        tmp_13 = in_5[(Ellipsis, slice(None, 128, None))]
        tmp_14 = in_5[(Ellipsis, slice(128, None, None))];  in_5 = None
        tmp_15 = -tmp_14;  tmp_14 = None
        tmp_16 = torch.cat((tmp_15, tmp_13), dim = -1);  tmp_15 = tmp_13 = None
        tmp_17 = tmp_16 * tmp_11;  tmp_16 = None
        tmp_18 = tmp_12 + tmp_17;  tmp_12 = tmp_17 = None
        tmp_19 = in_4 * tmp_10;  tmp_10 = None
        tmp_20 = in_4[(Ellipsis, slice(None, 128, None))]
        tmp_21 = in_4[(Ellipsis, slice(128, None, None))];  in_4 = None
        tmp_22 = -tmp_21;  tmp_21 = None
        tmp_23 = torch.cat((tmp_22, tmp_20), dim = -1);  tmp_22 = tmp_20 = None
        tmp_24 = tmp_23 * tmp_11;  tmp_23 = tmp_11 = None
        tmp_25 = tmp_19 + tmp_24;  tmp_19 = tmp_24 = None
        tmp_26 = tmp_25[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None))]
        tmp_27 = tmp_26.expand(1, 1, 8, 3, 256);  tmp_26 = None
        tmp_28 = tmp_27.reshape(1, 8, 3, 256);  tmp_27 = None
        tmp_29 = tmp_9[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None))]
        tmp_30 = tmp_29.expand(1, 1, 8, 3, 256);  tmp_29 = None
        tmp_31 = tmp_30.reshape(1, 8, 3, 256);  tmp_30 = None
        tmp_32 = in_0[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 3, None))];  in_0 = None
        tmp_33 = tmp_18.contiguous();  tmp_18 = None
        tmp_34 = tmp_28.contiguous();  tmp_28 = None
        tmp_35 = tmp_31.contiguous();  tmp_31 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_33, tmp_34, tmp_35, attn_mask = tmp_32, dropout_p = 0.0, scale = 0.0625, is_causal = False);  tmp_33 = tmp_34 = tmp_35 = tmp_32 = None
        tmp_37 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_38 = tmp_37.contiguous();  tmp_37 = None
        tmp_39 = tmp_38.reshape(1, 3, -1);  tmp_38 = None
        tmp_40 = tmp_39.contiguous();  tmp_39 = None
        linear_1 = torch.nn.functional.linear(tmp_40, w_5, None);  tmp_40 = w_5 = None
        tmp_42 = in_2 + linear_1;  in_2 = linear_1 = None
        tmp_43 = tmp_42.float()
        tmp_44 = tmp_43.pow(2)
        tmp_45 = tmp_44.mean(-1, keepdim = True);  tmp_44 = None
        tmp_46 = tmp_45 + 1e-06;  tmp_45 = None
        tmp_47 = torch.rsqrt(tmp_46);  tmp_46 = None
        tmp_48 = tmp_43 * tmp_47;  tmp_43 = tmp_47 = None
        tmp_49 = w_4.float();  w_4 = None
        tmp_50 = 1.0 + tmp_49;  tmp_49 = None
        tmp_51 = tmp_48 * tmp_50;  tmp_48 = tmp_50 = None
        tmp_52 = tmp_51.type_as(tmp_42);  tmp_51 = None
        linear_2 = torch.nn.functional.linear(tmp_52, w_2, None);  w_2 = None
        tmp_54 = torch.nn.functional.gelu(linear_2, approximate = 'tanh');  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_52, w_3, None);  tmp_52 = w_3 = None
        tmp_56 = tmp_54 * linear_3;  tmp_54 = linear_3 = None
        linear_4 = torch.nn.functional.linear(tmp_56, w_1, None);  tmp_56 = w_1 = None
        tmp_58 = tmp_42 + linear_4;  tmp_42 = linear_4 = None
        tmp_59 = tmp_58.float()
        tmp_60 = tmp_59.pow(2)
        tmp_61 = tmp_60.mean(-1, keepdim = True);  tmp_60 = None
        tmp_62 = tmp_61 + 1e-06;  tmp_61 = None
        tmp_63 = torch.rsqrt(tmp_62);  tmp_62 = None
        tmp_64 = tmp_59 * tmp_63;  tmp_59 = tmp_63 = None
        tmp_65 = w_0.float();  w_0 = None
        tmp_66 = 1.0 + tmp_65;  tmp_65 = None
        tmp_67 = tmp_64 * tmp_66;  tmp_64 = tmp_66 = None
        tmp_68 = tmp_67.type_as(tmp_58);  tmp_67 = None
        return (tmp_58, tmp_68, tmp_25, tmp_9)
        
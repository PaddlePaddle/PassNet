import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16):
        linear = torch.nn.functional.linear(in_13, in_7, in_6);  in_7 = in_6 = None
        tmp_11 = linear.view((1, 128, -1, 80));  linear = None
        tmp_12 = tmp_11.transpose(1, 2);  tmp_11 = None
        tmp_13 = in_15[(Ellipsis, slice(None, 32, None))]
        tmp_14 = in_15[(Ellipsis, slice(32, None, None))];  in_15 = None
        tmp_15 = in_14[(Ellipsis, slice(None, 32, None))]
        tmp_16 = in_14[(Ellipsis, slice(32, None, None))];  in_14 = None
        tmp_17 = in_11.unsqueeze(1);  in_11 = None
        tmp_18 = in_16.unsqueeze(1);  in_16 = None
        tmp_19 = tmp_13 * tmp_17
        tmp_20 = tmp_13[(Ellipsis, slice(None, 16, None))]
        tmp_21 = tmp_13[(Ellipsis, slice(16, None, None))];  tmp_13 = None
        tmp_22 = -tmp_21;  tmp_21 = None
        tmp_23 = torch.cat((tmp_22, tmp_20), dim = -1);  tmp_22 = tmp_20 = None
        tmp_24 = tmp_23 * tmp_18;  tmp_23 = None
        tmp_25 = tmp_19 + tmp_24;  tmp_19 = tmp_24 = None
        tmp_26 = tmp_15 * tmp_17;  tmp_17 = None
        tmp_27 = tmp_15[(Ellipsis, slice(None, 16, None))]
        tmp_28 = tmp_15[(Ellipsis, slice(16, None, None))];  tmp_15 = None
        tmp_29 = -tmp_28;  tmp_28 = None
        tmp_30 = torch.cat((tmp_29, tmp_27), dim = -1);  tmp_29 = tmp_27 = None
        tmp_31 = tmp_30 * tmp_18;  tmp_30 = tmp_18 = None
        tmp_32 = tmp_26 + tmp_31;  tmp_26 = tmp_31 = None
        tmp_33 = torch.cat((tmp_25, tmp_14), dim = -1);  tmp_25 = tmp_14 = None
        tmp_34 = torch.cat((tmp_32, tmp_16), dim = -1);  tmp_32 = tmp_16 = None
        tmp_35 = in_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))];  in_10 = None
        tmp_36 = tmp_33.contiguous();  tmp_33 = None
        tmp_37 = tmp_34.contiguous()
        tmp_38 = tmp_12.contiguous()
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_36, tmp_37, tmp_38, attn_mask = tmp_35, dropout_p = 0.0, scale = 0.11180339887498948, is_causal = False);  tmp_36 = tmp_37 = tmp_38 = tmp_35 = None
        tmp_40 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_41 = tmp_40.contiguous();  tmp_40 = None
        tmp_42 = tmp_41.reshape(1, 128, -1);  tmp_41 = None
        tmp_43 = tmp_42.contiguous();  tmp_42 = None
        linear_1 = torch.nn.functional.linear(tmp_43, in_5, in_4);  tmp_43 = in_5 = in_4 = None
        tmp_45 = torch.nn.functional.dropout(linear_1, 0.1, False, False);  linear_1 = None
        linear_2 = torch.nn.functional.linear(in_13, in_1, in_0);  in_13 = in_1 = in_0 = None
        tmp_47 = 0.5 * linear_2
        tmp_48 = torch.pow(linear_2, 3.0)
        tmp_49 = 0.044715 * tmp_48;  tmp_48 = None
        tmp_50 = linear_2 + tmp_49;  linear_2 = tmp_49 = None
        tmp_51 = 0.7978845608028654 * tmp_50;  tmp_50 = None
        tmp_52 = torch.tanh(tmp_51);  tmp_51 = None
        tmp_53 = 1.0 + tmp_52;  tmp_52 = None
        tmp_54 = tmp_47 * tmp_53;  tmp_47 = tmp_53 = None
        linear_3 = torch.nn.functional.linear(tmp_54, in_3, in_2);  tmp_54 = in_3 = in_2 = None
        tmp_56 = torch.nn.functional.dropout(linear_3, 0.1, False, False);  linear_3 = None
        tmp_57 = tmp_45 + tmp_56;  tmp_45 = tmp_56 = None
        tmp_58 = tmp_57 + in_12;  tmp_57 = in_12 = None
        tmp_59 = torch.nn.functional.layer_norm(tmp_58, (2560,), in_9, in_8, 1e-05);  in_9 = in_8 = None
        return (tmp_58, tmp_59, tmp_34, tmp_12)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, in_0, in_1, in_2, in_3, in_4):
        linear = torch.nn.functional.linear(in_2, w_3, w_2);  in_2 = w_3 = w_2 = None
        tmp_25 = linear.view(1, -1, 16, 64);  linear = None
        tmp_26 = tmp_25.transpose(1, 2);  tmp_25 = None
        tmp_27 = tmp_26.contiguous();  tmp_26 = None
        tmp_28 = in_1.contiguous();  in_1 = None
        tmp_29 = in_4.contiguous();  in_4 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_27, tmp_28, tmp_29, attn_mask = None, dropout_p = 0.0, scale = 0.125, is_causal = False);  tmp_27 = tmp_28 = tmp_29 = None
        tmp_31 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_32 = tmp_31.contiguous();  tmp_31 = None
        tmp_33 = tmp_32.reshape((1, 577, 1024));  tmp_32 = None
        linear_1 = torch.nn.functional.linear(tmp_33, w_5, w_4);  tmp_33 = w_5 = w_4 = None
        tmp_35 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_36 = tmp_35 * w_6;  tmp_35 = w_6 = None
        tmp_37 = tmp_36 + in_3;  tmp_36 = in_3 = None
        tmp_38 = torch.nn.functional.layer_norm(tmp_37, (1024,), w_13, w_12, 1e-06);  w_13 = w_12 = None
        linear_2 = torch.nn.functional.linear(tmp_38, w_9, w_8);  tmp_38 = w_9 = w_8 = None
        tmp_40 = torch.nn.functional.gelu(linear_2);  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_40, w_11, w_10);  tmp_40 = w_11 = w_10 = None
        tmp_42 = linear_3 * w_7;  linear_3 = w_7 = None
        tmp_43 = tmp_42 + tmp_37;  tmp_42 = tmp_37 = None
        tmp_44 = torch.nn.functional.layer_norm(tmp_43, (1024,), w_15, w_14, 1e-06);  tmp_43 = w_15 = w_14 = None
        linear_4 = torch.nn.functional.linear(tmp_44, w_17, w_16);  tmp_44 = w_17 = w_16 = None
        tmp_46 = linear_4[(slice(None, None, None), slice(-576, None, None), slice(None, None, None))];  linear_4 = None
        tmp_47 = tmp_46.reshape(1, 24, 24, 128);  tmp_46 = None
        tmp_48 = tmp_47.permute(0, 3, 1, 2);  tmp_47 = None
        tmp_49 = torch.nn.functional.interpolate(tmp_48, size = (24, 24), mode = 'bilinear', align_corners = False);  tmp_48 = None
        conv2d = torch.conv2d(in_0, w_1, w_0, (2, 2), (1, 1), (1, 1), 1);  in_0 = w_1 = w_0 = None
        tmp_51 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        tmp_52 = tmp_49 + tmp_51;  tmp_49 = tmp_51 = None
        tmp_53 = torch.nn.functional.interpolate(tmp_52, size = (24, 24), mode = 'bilinear', align_corners = False);  tmp_52 = None
        conv2d_1 = torch.conv2d(tmp_53, w_19, w_18, (2, 2), (1, 1), (1, 1), 1);  tmp_53 = w_19 = w_18 = None
        tmp_55 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        conv2d_2 = torch.conv2d(tmp_55, w_21, w_20, (2, 2), (1, 1), (1, 1), 1);  tmp_55 = w_21 = w_20 = None
        tmp_57 = torch.nn.functional.relu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_57, w_23, w_22, (1, 1), (0, 0), (1, 1), 1);  tmp_57 = w_23 = w_22 = None
        tmp_59 = conv2d_3.flatten();  conv2d_3 = None
        return (tmp_59,)
        
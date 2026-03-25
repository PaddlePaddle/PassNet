import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, in_0, in_1, in_2, in_3, in_4):
        tmp_32 = torch.nn.functional.gelu(in_0, approximate = 'none');  in_0 = None
        tmp_33 = torch.nn.functional.dropout(tmp_32, 0.0, False, False);  tmp_32 = None
        linear = torch.nn.functional.linear(tmp_33, w_1, w_0);  tmp_33 = w_1 = w_0 = None
        tmp_35 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_36 = in_2 + tmp_35;  in_2 = tmp_35 = None
        tmp_37 = torch.nn.functional.layer_norm(tmp_36, (512,), w_11, w_10, 1e-05);  w_11 = w_10 = None
        tmp_38 = tmp_37.transpose(0, 1)
        tmp_39 = tmp_37.transpose(0, 1);  tmp_37 = None
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(tmp_38, tmp_39, tmp_39, 512, 16, w_5, w_4, None, None, False, 0.0, w_3, w_2, training = False, key_padding_mask = None, need_weights = True, attn_mask = None, average_attn_weights = True, is_causal = False);  tmp_38 = tmp_39 = w_5 = w_4 = w_3 = w_2 = None
        tmp_41 = multi_head_attention_forward[0];  multi_head_attention_forward = None
        tmp_42 = tmp_41.transpose(0, 1);  tmp_41 = None
        tmp_43 = torch.nn.functional.dropout(tmp_42, 0.0, False, False);  tmp_42 = None
        tmp_44 = 0.0 + tmp_43;  tmp_43 = None
        tmp_45 = tmp_36 + tmp_44;  tmp_36 = tmp_44 = None
        tmp_46 = torch.nn.functional.layer_norm(tmp_45, (512,), w_13, w_12, 1e-05);  w_13 = w_12 = None
        linear_1 = torch.nn.functional.linear(tmp_46, w_7, w_6);  tmp_46 = w_7 = w_6 = None
        tmp_48 = torch.nn.functional.gelu(linear_1, approximate = 'none');  linear_1 = None
        tmp_49 = torch.nn.functional.dropout(tmp_48, 0.0, False, False);  tmp_48 = None
        linear_2 = torch.nn.functional.linear(tmp_49, w_9, w_8);  tmp_49 = w_9 = w_8 = None
        tmp_51 = torch.nn.functional.dropout(linear_2, 0.0, False, False);  linear_2 = None
        tmp_52 = tmp_45 + tmp_51;  tmp_45 = tmp_51 = None
        tmp_53 = torch.nn.functional.layer_norm(tmp_52, (512,), w_15, w_14, 1e-05);  tmp_52 = w_15 = w_14 = None
        tmp_54 = tmp_53.reshape(1, 16, 16, -1);  tmp_53 = None
        tmp_55 = tmp_54.permute(0, 3, 1, 2);  tmp_54 = None
        tmp_56 = tmp_55.contiguous();  tmp_55 = None
        conv2d = torch.conv2d(in_3, w_25, w_24, (1, 1), (0, 0), (1, 1), 1);  in_3 = w_25 = w_24 = None
        conv2d_1 = torch.conv2d(in_4, w_27, w_26, (1, 1), (0, 0), (1, 1), 1);  in_4 = w_27 = w_26 = None
        conv2d_2 = torch.conv2d(in_1, w_29, w_28, (1, 1), (0, 0), (1, 1), 1);  in_1 = w_29 = w_28 = None
        conv2d_3 = torch.conv2d(tmp_56, w_31, w_30, (1, 1), (0, 0), (1, 1), 1);  tmp_56 = w_31 = w_30 = None
        tmp_61 = torch.nn.functional.interpolate(conv2d_3, (32, 32), None, 'nearest', None)
        tmp_62 = conv2d_2 + tmp_61;  conv2d_2 = tmp_61 = None
        tmp_63 = torch.nn.functional.interpolate(tmp_62, (64, 64), None, 'nearest', None)
        tmp_64 = conv2d_1 + tmp_63;  conv2d_1 = tmp_63 = None
        tmp_65 = torch.nn.functional.interpolate(tmp_64, (128, 128), None, 'nearest', None)
        tmp_66 = conv2d + tmp_65;  conv2d = tmp_65 = None
        conv2d_4 = torch.conv2d(tmp_66, w_17, w_16, (1, 1), (1, 1), (1, 1), 1);  tmp_66 = w_17 = w_16 = None
        conv2d_5 = torch.conv2d(tmp_64, w_19, w_18, (1, 1), (1, 1), (1, 1), 1);  tmp_64 = w_19 = w_18 = None
        conv2d_6 = torch.conv2d(tmp_62, w_21, w_20, (1, 1), (1, 1), (1, 1), 1);  tmp_62 = w_21 = w_20 = None
        conv2d_7 = torch.conv2d(conv2d_3, w_23, w_22, (1, 1), (1, 1), (1, 1), 1);  conv2d_3 = w_23 = w_22 = None
        return (conv2d_4, conv2d_5, conv2d_6, conv2d_7)
        
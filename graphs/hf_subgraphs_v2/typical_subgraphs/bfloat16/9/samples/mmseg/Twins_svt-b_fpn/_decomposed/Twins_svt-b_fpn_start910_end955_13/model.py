import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, in_0, in_1, in_2, in_3, in_4):
        tmp_34 = torch.nn.functional.gelu(in_0, approximate = 'none');  in_0 = None
        tmp_35 = torch.nn.functional.dropout(tmp_34, 0.0, False, False);  tmp_34 = None
        linear = torch.nn.functional.linear(tmp_35, w_1, w_0);  tmp_35 = w_1 = w_0 = None
        tmp_37 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_38 = in_2 + tmp_37;  in_2 = tmp_37 = None
        tmp_39 = tmp_38.transpose(1, 2);  tmp_38 = None
        tmp_40 = tmp_39.view(1, 768, 16, 16);  tmp_39 = None
        conv2d = torch.conv2d(tmp_40, w_17, w_16, (1, 1), (1, 1), (1, 1), 768);  w_17 = w_16 = None
        tmp_42 = conv2d + tmp_40;  conv2d = tmp_40 = None
        tmp_43 = tmp_42.flatten(2);  tmp_42 = None
        tmp_44 = tmp_43.transpose(1, 2);  tmp_43 = None
        tmp_45 = torch.nn.functional.layer_norm(tmp_44, (768,), w_11, w_10, 1e-05);  w_11 = w_10 = None
        tmp_46 = tmp_45.transpose(0, 1)
        tmp_47 = tmp_45.transpose(0, 1);  tmp_45 = None
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(tmp_46, tmp_47, tmp_47, 768, 24, w_5, w_4, None, None, False, 0.0, w_3, w_2, training = False, key_padding_mask = None, need_weights = True, attn_mask = None, average_attn_weights = True, is_causal = False);  tmp_46 = tmp_47 = w_5 = w_4 = w_3 = w_2 = None
        tmp_49 = multi_head_attention_forward[0];  multi_head_attention_forward = None
        tmp_50 = tmp_49.transpose(0, 1);  tmp_49 = None
        tmp_51 = torch.nn.functional.dropout(tmp_50, 0.0, False, False);  tmp_50 = None
        tmp_52 = 0.0 + tmp_51;  tmp_51 = None
        tmp_53 = tmp_44 + tmp_52;  tmp_44 = tmp_52 = None
        tmp_54 = torch.nn.functional.layer_norm(tmp_53, (768,), w_13, w_12, 1e-05);  w_13 = w_12 = None
        linear_1 = torch.nn.functional.linear(tmp_54, w_7, w_6);  tmp_54 = w_7 = w_6 = None
        tmp_56 = torch.nn.functional.gelu(linear_1, approximate = 'none');  linear_1 = None
        tmp_57 = torch.nn.functional.dropout(tmp_56, 0.0, False, False);  tmp_56 = None
        linear_2 = torch.nn.functional.linear(tmp_57, w_9, w_8);  tmp_57 = w_9 = w_8 = None
        tmp_59 = torch.nn.functional.dropout(linear_2, 0.0, False, False);  linear_2 = None
        tmp_60 = tmp_53 + tmp_59;  tmp_53 = tmp_59 = None
        tmp_61 = torch.nn.functional.layer_norm(tmp_60, (768,), w_15, w_14, 1e-05);  tmp_60 = w_15 = w_14 = None
        tmp_62 = tmp_61.reshape(1, 16, 16, -1);  tmp_61 = None
        tmp_63 = tmp_62.permute(0, 3, 1, 2);  tmp_62 = None
        tmp_64 = tmp_63.contiguous();  tmp_63 = None
        conv2d_1 = torch.conv2d(in_3, w_27, w_26, (1, 1), (0, 0), (1, 1), 1);  in_3 = w_27 = w_26 = None
        conv2d_2 = torch.conv2d(in_4, w_29, w_28, (1, 1), (0, 0), (1, 1), 1);  in_4 = w_29 = w_28 = None
        conv2d_3 = torch.conv2d(in_1, w_31, w_30, (1, 1), (0, 0), (1, 1), 1);  in_1 = w_31 = w_30 = None
        conv2d_4 = torch.conv2d(tmp_64, w_33, w_32, (1, 1), (0, 0), (1, 1), 1);  tmp_64 = w_33 = w_32 = None
        tmp_69 = torch.nn.functional.interpolate(conv2d_4, (32, 32), None, 'nearest', None)
        tmp_70 = conv2d_3 + tmp_69;  conv2d_3 = tmp_69 = None
        tmp_71 = torch.nn.functional.interpolate(tmp_70, (64, 64), None, 'nearest', None)
        tmp_72 = conv2d_2 + tmp_71;  conv2d_2 = tmp_71 = None
        tmp_73 = torch.nn.functional.interpolate(tmp_72, (128, 128), None, 'nearest', None)
        tmp_74 = conv2d_1 + tmp_73;  conv2d_1 = tmp_73 = None
        conv2d_5 = torch.conv2d(tmp_74, w_19, w_18, (1, 1), (1, 1), (1, 1), 1);  tmp_74 = w_19 = w_18 = None
        conv2d_6 = torch.conv2d(tmp_72, w_21, w_20, (1, 1), (1, 1), (1, 1), 1);  tmp_72 = w_21 = w_20 = None
        conv2d_7 = torch.conv2d(tmp_70, w_23, w_22, (1, 1), (1, 1), (1, 1), 1);  tmp_70 = w_23 = w_22 = None
        conv2d_8 = torch.conv2d(conv2d_4, w_25, w_24, (1, 1), (1, 1), (1, 1), 1);  conv2d_4 = w_25 = w_24 = None
        return (conv2d_5, conv2d_6, conv2d_7, conv2d_8)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, in_0, in_1, in_2, in_3):
        tmp_34 = torch.nn.functional.layer_norm(in_0, (512,), w_3, w_2, 1e-05);  in_0 = w_3 = w_2 = None
        tmp_35 = torch.nn.functional.relu(tmp_34, inplace = True);  tmp_34 = None
        linear = torch.nn.functional.linear(tmp_35, w_1, w_0);  tmp_35 = w_1 = w_0 = None
        tmp_37 = linear.permute(0, 1, 3, 2);  linear = None
        tmp_38 = tmp_37.reshape(1, 150, 512, 1, 1);  tmp_37 = None
        tmp_39 = in_2[slice(0, 1, None)];  in_2 = None
        tmp_40 = tmp_38[0];  tmp_38 = None
        conv2d = torch.conv2d(tmp_39, tmp_40, padding = 0);  tmp_39 = tmp_40 = None
        tmp_42 = torch.cat([conv2d], dim = 0);  conv2d = None
        tmp_43 = tmp_42.reshape(1, 150, 64, 64);  tmp_42 = None
        tmp_44 = in_1.permute(0, 1, 3, 2);  in_1 = None
        tmp_45 = tmp_44.reshape(1, 150, 512, 1, 1);  tmp_44 = None
        conv2d_1 = torch.conv2d(in_3, w_11, w_10, (1, 1), (0, 0), (1, 1), 1);  in_3 = w_11 = w_10 = None
        tmp_47 = tmp_43.softmax(dim = 1);  tmp_43 = None
        einsum = torch.functional.einsum('bnhw,bchw->bnc', tmp_47, conv2d_1);  tmp_47 = None
        tmp_49 = tmp_45.reshape(1, 150, 512, -1);  tmp_45 = None
        tmp_50 = tmp_49.permute(0, 1, 3, 2);  tmp_49 = None
        tmp_51 = einsum.reshape(-1, 256);  einsum = None
        linear_1 = torch.nn.functional.linear(tmp_51, w_15, w_14);  tmp_51 = w_15 = w_14 = None
        tmp_53 = linear_1[(slice(None, None, None), slice(None, 256, None))]
        tmp_54 = tmp_53.view(-1, 256);  tmp_53 = None
        tmp_55 = linear_1[(slice(None, None, None), slice(-256, None, None))];  linear_1 = None
        tmp_56 = tmp_55.view(-1, 256);  tmp_55 = None
        tmp_57 = tmp_50.reshape(300, -1, 256);  tmp_50 = None
        linear_2 = torch.nn.functional.linear(tmp_57, w_23, w_22);  tmp_57 = w_23 = w_22 = None
        tmp_59 = linear_2[(Ellipsis, slice(None, 256, None))]
        tmp_60 = linear_2[(Ellipsis, slice(-256, None, None))];  linear_2 = None
        tmp_61 = tmp_54.unsqueeze(-2);  tmp_54 = None
        tmp_62 = tmp_59 * tmp_61;  tmp_59 = tmp_61 = None
        linear_3 = torch.nn.functional.linear(tmp_62, w_21, w_20);  w_21 = w_20 = None
        tmp_64 = torch.nn.functional.layer_norm(linear_3, (256,), w_25, w_24, 1e-05);  linear_3 = w_25 = w_24 = None
        linear_4 = torch.nn.functional.linear(tmp_62, w_33, w_32);  tmp_62 = w_33 = w_32 = None
        tmp_66 = torch.nn.functional.layer_norm(linear_4, (256,), w_29, w_28, 1e-05);  linear_4 = w_29 = w_28 = None
        tmp_67 = tmp_64.sigmoid();  tmp_64 = None
        tmp_68 = tmp_66.sigmoid();  tmp_66 = None
        tmp_69 = torch.nn.functional.layer_norm(tmp_56, (256,), w_31, w_30, 1e-05);  tmp_56 = w_31 = w_30 = None
        tmp_70 = torch.nn.functional.layer_norm(tmp_60, (256,), w_27, w_26, 1e-05);  tmp_60 = w_27 = w_26 = None
        tmp_71 = tmp_69.unsqueeze(-2);  tmp_69 = None
        tmp_72 = tmp_68 * tmp_71;  tmp_68 = tmp_71 = None
        tmp_73 = tmp_67 * tmp_70;  tmp_67 = tmp_70 = None
        tmp_74 = tmp_72 + tmp_73;  tmp_72 = tmp_73 = None
        linear_5 = torch.nn.functional.linear(tmp_74, w_17, w_16);  tmp_74 = w_17 = w_16 = None
        tmp_76 = torch.nn.functional.layer_norm(linear_5, (256,), w_19, w_18, 1e-05);  linear_5 = w_19 = w_18 = None
        tmp_77 = torch.nn.functional.relu(tmp_76, inplace = True);  tmp_76 = None
        tmp_78 = tmp_77.reshape(1, 150, -1);  tmp_77 = None
        tmp_79 = tmp_78.permute(1, 0, 2);  tmp_78 = None
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(tmp_79, tmp_79, tmp_79, 512, 8, w_7, w_6, None, None, False, 0.0, w_5, w_4, training = False, key_padding_mask = None, need_weights = True, attn_mask = None, average_attn_weights = True, is_causal = False);  w_7 = w_6 = w_5 = w_4 = None
        tmp_81 = multi_head_attention_forward[0];  multi_head_attention_forward = None
        tmp_82 = torch.nn.functional.dropout(tmp_81, 0.0, False, False);  tmp_81 = None
        tmp_83 = torch.nn.functional.dropout(tmp_82, 0.0, False, False);  tmp_82 = None
        tmp_84 = tmp_79 + tmp_83;  tmp_79 = tmp_83 = None
        tmp_85 = torch.nn.functional.layer_norm(tmp_84, (512,), w_9, w_8, 1e-05);  tmp_84 = w_9 = w_8 = None
        tmp_86 = tmp_85.permute(1, 0, 2);  tmp_85 = None
        tmp_87 = tmp_86.reshape(1, 150, -1, 512);  tmp_86 = None
        linear_6 = torch.nn.functional.linear(tmp_87, w_13, w_12);  w_13 = w_12 = None
        tmp_89 = torch.nn.functional.relu(linear_6, inplace = True);  linear_6 = None
        tmp_90 = torch.nn.functional.dropout(tmp_89, 0.0, False, False);  tmp_89 = None
        return (tmp_90, tmp_87, conv2d_1)
        
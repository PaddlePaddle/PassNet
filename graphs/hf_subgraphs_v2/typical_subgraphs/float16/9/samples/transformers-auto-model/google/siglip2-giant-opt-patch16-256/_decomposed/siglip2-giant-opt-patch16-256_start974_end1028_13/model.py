import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, w_28 : torch.Tensor, w_29 : torch.Tensor, w_30 : torch.Tensor, w_31 : torch.Tensor, w_32 : torch.Tensor, w_33 : torch.Tensor, w_34 : torch.Tensor, w_35 : torch.Tensor, w_36 : torch.Tensor, w_37 : torch.Tensor, w_38 : torch.Tensor, w_39 : torch.Tensor, w_40 : torch.Tensor, w_41 : torch.Tensor, w_42 : torch.Tensor, w_43 : torch.Tensor, w_44 : torch.Tensor, w_45 : torch.Tensor, w_46 : torch.Tensor, w_47 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_1, w_18, w_17);  in_1 = w_18 = w_17 = None
        tmp_50 = in_2 + linear;  in_2 = linear = None
        tmp_51 = torch.nn.functional.layer_norm(tmp_50, (1536,), w_12, w_11, 1e-06);  w_12 = w_11 = None
        linear_1 = torch.nn.functional.linear(tmp_51, w_14, w_13);  tmp_51 = w_14 = w_13 = None
        tmp_53 = torch.nn.functional.gelu(linear_1, approximate = 'tanh');  linear_1 = None
        linear_2 = torch.nn.functional.linear(tmp_53, w_16, w_15);  tmp_53 = w_16 = w_15 = None
        tmp_55 = tmp_50 + linear_2;  tmp_50 = linear_2 = None
        tmp_56 = torch.nn.functional.layer_norm(tmp_55, (1536,), w_20, w_19, 1e-06);  w_20 = w_19 = None
        linear_3 = torch.nn.functional.linear(tmp_56, w_32, w_31);  w_32 = w_31 = None
        linear_4 = torch.nn.functional.linear(tmp_56, w_28, w_27);  w_28 = w_27 = None
        linear_5 = torch.nn.functional.linear(tmp_56, w_34, w_33);  tmp_56 = w_34 = w_33 = None
        tmp_60 = linear_3.view(1, 256, 16, 96);  linear_3 = None
        tmp_61 = tmp_60.transpose(1, 2);  tmp_60 = None
        tmp_62 = linear_4.view(1, 256, 16, 96);  linear_4 = None
        tmp_63 = tmp_62.transpose(1, 2);  tmp_62 = None
        tmp_64 = linear_5.view(1, 256, 16, 96);  linear_5 = None
        tmp_65 = tmp_64.transpose(1, 2);  tmp_64 = None
        tmp_66 = tmp_61.contiguous();  tmp_61 = None
        tmp_67 = tmp_63.contiguous();  tmp_63 = None
        tmp_68 = tmp_65.contiguous();  tmp_65 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_66, tmp_67, tmp_68, attn_mask = None, dropout_p = 0.0, scale = 0.10206207261596575, is_causal = False);  tmp_66 = tmp_67 = tmp_68 = None
        tmp_70 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_71 = tmp_70.contiguous();  tmp_70 = None
        tmp_72 = tmp_71.reshape(1, 256, 1536);  tmp_71 = None
        tmp_73 = tmp_72.contiguous();  tmp_72 = None
        linear_6 = torch.nn.functional.linear(tmp_73, w_30, w_29);  tmp_73 = w_30 = w_29 = None
        tmp_75 = tmp_55 + linear_6;  tmp_55 = linear_6 = None
        tmp_76 = torch.nn.functional.layer_norm(tmp_75, (1536,), w_22, w_21, 1e-06);  w_22 = w_21 = None
        linear_7 = torch.nn.functional.linear(tmp_76, w_24, w_23);  tmp_76 = w_24 = w_23 = None
        tmp_78 = torch.nn.functional.gelu(linear_7, approximate = 'tanh');  linear_7 = None
        linear_8 = torch.nn.functional.linear(tmp_78, w_26, w_25);  tmp_78 = w_26 = w_25 = None
        tmp_80 = tmp_75 + linear_8;  tmp_75 = linear_8 = None
        tmp_81 = torch.nn.functional.layer_norm(tmp_80, (1536,), w_47, w_46, 1e-06);  tmp_80 = w_47 = w_46 = None
        tmp_82 = w_45.repeat(1, 1, 1);  w_45 = None
        tmp_83 = tmp_82.transpose(1, 0);  tmp_82 = None
        tmp_84 = tmp_81.transpose(1, 0)
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(tmp_83, tmp_84, tmp_84, 1536, 16, w_38, w_37, None, None, False, 0.0, w_36, w_35, training = False, key_padding_mask = None, need_weights = True, attn_mask = None, average_attn_weights = True, is_causal = False);  tmp_83 = tmp_84 = w_38 = w_37 = w_36 = w_35 = None
        tmp_86 = multi_head_attention_forward[0];  multi_head_attention_forward = None
        tmp_87 = tmp_86.transpose(1, 0);  tmp_86 = None
        tmp_88 = torch.nn.functional.layer_norm(tmp_87, (1536,), w_40, w_39, 1e-06);  w_40 = w_39 = None
        linear_9 = torch.nn.functional.linear(tmp_88, w_42, w_41);  tmp_88 = w_42 = w_41 = None
        tmp_90 = torch.nn.functional.gelu(linear_9, approximate = 'tanh');  linear_9 = None
        linear_10 = torch.nn.functional.linear(tmp_90, w_44, w_43);  tmp_90 = w_44 = w_43 = None
        tmp_92 = tmp_87 + linear_10;  tmp_87 = linear_10 = None
        tmp_93 = tmp_92[(slice(None, None, None), 0)];  tmp_92 = None
        tmp_94 = in_0.view(-1, 7);  in_0 = None
        tmp_95 = w_0[(slice(None, None, None), slice(None, 7, None))];  w_0 = None
        tmp_96 = torch.nn.functional.embedding(tmp_94, w_2, None, None, 2.0, False, False);  tmp_94 = w_2 = None
        tmp_97 = torch.nn.functional.embedding(tmp_95, w_1, None, None, 2.0, False, False);  tmp_95 = w_1 = None
        tmp_98 = tmp_96 + tmp_97;  tmp_96 = tmp_97 = None
        tmp_99 = torch.nn.functional.layer_norm(tmp_98, (1152,), w_4, w_3, 1e-06);  w_4 = w_3 = None
        linear_11 = torch.nn.functional.linear(tmp_99, w_8, w_7);  w_8 = w_7 = None
        linear_12 = torch.nn.functional.linear(tmp_99, w_6, w_5);  w_6 = w_5 = None
        linear_13 = torch.nn.functional.linear(tmp_99, w_10, w_9);  tmp_99 = w_10 = w_9 = None
        return (tmp_98, linear_12, tmp_81, tmp_93, linear_11, linear_13)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor, in_29 : torch.Tensor, in_30 : torch.Tensor, in_31 : torch.Tensor, in_32 : torch.Tensor, in_33 : torch.Tensor, in_34 : torch.Tensor, in_35 : torch.Tensor, in_36 : torch.Tensor, in_37 : torch.Tensor, in_38 : torch.Tensor, in_39 : torch.Tensor, in_40 : torch.Tensor, in_41 : torch.Tensor, in_42 : torch.Tensor, in_43 : torch.Tensor, in_44 : torch.Tensor, in_45 : torch.Tensor, in_46 : torch.Tensor, in_47 : torch.Tensor, in_48 : torch.Tensor, in_49 : torch.Tensor, in_50 : torch.Tensor):
        linear = torch.nn.functional.linear(in_49, in_19, in_18);  in_49 = in_19 = in_18 = None
        tmp_50 = in_50 + linear;  in_50 = linear = None
        tmp_51 = torch.nn.functional.layer_norm(tmp_50, (1536,), in_13, in_12, 1e-06);  in_13 = in_12 = None
        to = tmp_51.to(torch.float16);  tmp_51 = None
        linear_1 = torch.nn.functional.linear(to, in_15, in_14);  to = in_15 = in_14 = None
        tmp_53 = torch.nn.functional.gelu(linear_1, approximate = 'tanh');  linear_1 = None
        to_1 = tmp_53.to(torch.float16);  tmp_53 = None
        linear_2 = torch.nn.functional.linear(to_1, in_17, in_16);  to_1 = in_17 = in_16 = None
        tmp_55 = tmp_50 + linear_2;  tmp_50 = linear_2 = None
        tmp_56 = torch.nn.functional.layer_norm(tmp_55, (1536,), in_21, in_20, 1e-06);  in_21 = in_20 = None
        to_2 = tmp_56.to(torch.float16)
        linear_3 = torch.nn.functional.linear(to_2, in_33, in_32);  to_2 = in_33 = in_32 = None
        to_3 = tmp_56.to(torch.float16)
        linear_4 = torch.nn.functional.linear(to_3, in_29, in_28);  to_3 = in_29 = in_28 = None
        to_4 = tmp_56.to(torch.float16);  tmp_56 = None
        linear_5 = torch.nn.functional.linear(to_4, in_35, in_34);  to_4 = in_35 = in_34 = None
        tmp_60 = linear_3.view(1, 576, 16, 96);  linear_3 = None
        tmp_61 = tmp_60.transpose(1, 2);  tmp_60 = None
        tmp_62 = linear_4.view(1, 576, 16, 96);  linear_4 = None
        tmp_63 = tmp_62.transpose(1, 2);  tmp_62 = None
        tmp_64 = linear_5.view(1, 576, 16, 96);  linear_5 = None
        tmp_65 = tmp_64.transpose(1, 2);  tmp_64 = None
        tmp_66 = tmp_61.contiguous();  tmp_61 = None
        tmp_67 = tmp_63.contiguous();  tmp_63 = None
        tmp_68 = tmp_65.contiguous();  tmp_65 = None
        to_5 = tmp_66.to(torch.float16);  tmp_66 = None
        to_6 = tmp_67.to(torch.float16);  tmp_67 = None
        to_7 = tmp_68.to(torch.float16);  tmp_68 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(to_5, to_6, to_7, attn_mask = None, dropout_p = 0.0, scale = 0.10206207261596575, is_causal = False);  to_5 = to_6 = to_7 = None
        tmp_70 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_71 = tmp_70.contiguous();  tmp_70 = None
        tmp_72 = tmp_71.reshape(1, 576, 1536);  tmp_71 = None
        tmp_73 = tmp_72.contiguous();  tmp_72 = None
        to_8 = tmp_73.to(torch.float16);  tmp_73 = None
        linear_6 = torch.nn.functional.linear(to_8, in_31, in_30);  to_8 = in_31 = in_30 = None
        tmp_75 = tmp_55 + linear_6;  tmp_55 = linear_6 = None
        tmp_76 = torch.nn.functional.layer_norm(tmp_75, (1536,), in_23, in_22, 1e-06);  in_23 = in_22 = None
        to_9 = tmp_76.to(torch.float16);  tmp_76 = None
        linear_7 = torch.nn.functional.linear(to_9, in_25, in_24);  to_9 = in_25 = in_24 = None
        tmp_78 = torch.nn.functional.gelu(linear_7, approximate = 'tanh');  linear_7 = None
        to_10 = tmp_78.to(torch.float16);  tmp_78 = None
        linear_8 = torch.nn.functional.linear(to_10, in_27, in_26);  to_10 = in_27 = in_26 = None
        tmp_80 = tmp_75 + linear_8;  tmp_75 = linear_8 = None
        tmp_81 = torch.nn.functional.layer_norm(tmp_80, (1536,), in_48, in_47, 1e-06);  tmp_80 = in_48 = in_47 = None
        tmp_82 = in_46.repeat(1, 1, 1);  in_46 = None
        tmp_83 = tmp_82.transpose(1, 0);  tmp_82 = None
        tmp_84 = tmp_81.transpose(1, 0)
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(tmp_83, tmp_84, tmp_84, 1536, 16, in_39, in_38, None, None, False, 0.0, in_37, in_36, training = False, key_padding_mask = None, need_weights = True, attn_mask = None, average_attn_weights = True, is_causal = False);  tmp_83 = tmp_84 = in_39 = in_38 = in_37 = in_36 = None
        tmp_86 = multi_head_attention_forward[0];  multi_head_attention_forward = None
        tmp_87 = tmp_86.transpose(1, 0);  tmp_86 = None
        tmp_88 = torch.nn.functional.layer_norm(tmp_87, (1536,), in_41, in_40, 1e-06);  in_41 = in_40 = None
        to_11 = tmp_88.to(torch.float16);  tmp_88 = None
        linear_9 = torch.nn.functional.linear(to_11, in_43, in_42);  to_11 = in_43 = in_42 = None
        tmp_90 = torch.nn.functional.gelu(linear_9, approximate = 'tanh');  linear_9 = None
        to_12 = tmp_90.to(torch.float16);  tmp_90 = None
        linear_10 = torch.nn.functional.linear(to_12, in_45, in_44);  to_12 = in_45 = in_44 = None
        tmp_92 = tmp_87 + linear_10;  tmp_87 = linear_10 = None
        tmp_93 = tmp_92[(slice(None, None, None), 0)];  tmp_92 = None
        tmp_94 = in_0.view(-1, 7);  in_0 = None
        tmp_95 = in_1[(slice(None, None, None), slice(None, 7, None))];  in_1 = None
        tmp_96 = torch.nn.functional.embedding(tmp_94, in_3, None, None, 2.0, False, False);  tmp_94 = in_3 = None
        tmp_97 = torch.nn.functional.embedding(tmp_95, in_2, None, None, 2.0, False, False);  tmp_95 = in_2 = None
        tmp_98 = tmp_96 + tmp_97;  tmp_96 = tmp_97 = None
        tmp_99 = torch.nn.functional.layer_norm(tmp_98, (1152,), in_5, in_4, 1e-06);  in_5 = in_4 = None
        to_13 = tmp_99.to(torch.float16)
        linear_11 = torch.nn.functional.linear(to_13, in_9, in_8);  to_13 = in_9 = in_8 = None
        to_14 = tmp_99.to(torch.float16)
        linear_12 = torch.nn.functional.linear(to_14, in_7, in_6);  to_14 = in_7 = in_6 = None
        to_15 = tmp_99.to(torch.float16);  tmp_99 = None
        linear_13 = torch.nn.functional.linear(to_15, in_11, in_10);  to_15 = in_11 = in_10 = None
        return (tmp_98, linear_12, tmp_81, tmp_93, linear_11, linear_13)
        
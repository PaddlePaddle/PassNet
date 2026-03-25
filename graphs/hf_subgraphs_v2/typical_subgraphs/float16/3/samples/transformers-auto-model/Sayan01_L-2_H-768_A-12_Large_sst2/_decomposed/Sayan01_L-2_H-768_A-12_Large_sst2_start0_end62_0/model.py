import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor, in_29 : torch.Tensor, in_30 : torch.Tensor, in_31 : torch.Tensor, in_32 : torch.Tensor, in_33 : torch.Tensor, in_34 : torch.Tensor, in_35 : torch.Tensor, in_36 : torch.Tensor, in_37 : torch.Tensor, in_38 : torch.Tensor, in_39 : torch.Tensor, in_40 : torch.Tensor, in_41 : torch.Tensor, in_42 : torch.Tensor):
        tmp_43 = in_2[(slice(None, None, None), slice(0, 64, None))];  in_2 = None
        tmp_44 = torch.nn.functional.embedding(in_1, in_7, 0, None, 2.0, False, False);  in_1 = in_7 = None
        tmp_45 = torch.nn.functional.embedding(in_42, in_6, None, None, 2.0, False, False);  in_42 = in_6 = None
        tmp_46 = tmp_44 + tmp_45;  tmp_44 = tmp_45 = None
        tmp_47 = torch.nn.functional.embedding(tmp_43, in_5, None, None, 2.0, False, False);  tmp_43 = in_5 = None
        tmp_46 += tmp_47;  tmp_48 = tmp_46;  tmp_46 = tmp_47 = None
        tmp_49 = torch.nn.functional.layer_norm(tmp_48, (768,), in_4, in_3, 1e-12);  tmp_48 = in_4 = in_3 = None
        tmp_50 = torch.nn.functional.dropout(tmp_49, 0.1, False, False);  tmp_49 = None
        tmp_51 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_52 = tmp_51.expand(32, 1, 64, 64);  tmp_51 = None
        tmp_53 = tmp_52.to(torch.float32);  tmp_52 = None
        tmp_54 = torch.tensor(1.0, dtype = torch.float32)
        tmp_55 = tmp_54 - tmp_53;  tmp_54 = tmp_53 = None
        tmp_56 = tmp_55.to(torch.bool)
        tmp_57 = tmp_55.masked_fill(tmp_56, -3.4028234663852886e+38);  tmp_55 = tmp_56 = None
        linear = torch.nn.functional.linear(tmp_50, in_15, in_14);  in_15 = in_14 = None
        tmp_59 = linear.view(32, -1, 12, 64);  linear = None
        tmp_60 = tmp_59.transpose(1, 2);  tmp_59 = None
        linear_1 = torch.nn.functional.linear(tmp_50, in_13, in_12);  in_13 = in_12 = None
        tmp_62 = linear_1.view(32, -1, 12, 64);  linear_1 = None
        tmp_63 = tmp_62.transpose(1, 2);  tmp_62 = None
        linear_2 = torch.nn.functional.linear(tmp_50, in_17, in_16);  in_17 = in_16 = None
        tmp_65 = linear_2.view(32, -1, 12, 64);  linear_2 = None
        tmp_66 = tmp_65.transpose(1, 2);  tmp_65 = None
        to_6 = tmp_57.to(torch.float16)
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_60, tmp_63, tmp_66, attn_mask = to_6, dropout_p = 0.0, is_causal = False);  tmp_60 = tmp_63 = tmp_66 = to_6 = None
        tmp_68 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_69 = tmp_68.reshape(32, 64, 768);  tmp_68 = None
        linear_3 = torch.nn.functional.linear(tmp_69, in_11, in_10);  tmp_69 = in_11 = in_10 = None
        tmp_71 = torch.nn.functional.dropout(linear_3, 0.1, False, False);  linear_3 = None
        tmp_72 = tmp_71 + tmp_50;  tmp_71 = tmp_50 = None
        tmp_73 = torch.nn.functional.layer_norm(tmp_72, (768,), in_9, in_8, 1e-12);  tmp_72 = in_9 = in_8 = None
        linear_4 = torch.nn.functional.linear(tmp_73, in_19, in_18);  in_19 = in_18 = None
        tmp_75 = torch.nn.functional.gelu(linear_4);  linear_4 = None
        linear_5 = torch.nn.functional.linear(tmp_75, in_23, in_22);  tmp_75 = in_23 = in_22 = None
        tmp_77 = torch.nn.functional.dropout(linear_5, 0.1, False, False);  linear_5 = None
        tmp_78 = tmp_77 + tmp_73;  tmp_77 = tmp_73 = None
        tmp_79 = torch.nn.functional.layer_norm(tmp_78, (768,), in_21, in_20, 1e-12);  tmp_78 = in_21 = in_20 = None
        linear_6 = torch.nn.functional.linear(tmp_79, in_31, in_30);  in_31 = in_30 = None
        tmp_81 = linear_6.view(32, -1, 12, 64);  linear_6 = None
        tmp_82 = tmp_81.transpose(1, 2);  tmp_81 = None
        linear_7 = torch.nn.functional.linear(tmp_79, in_29, in_28);  in_29 = in_28 = None
        tmp_84 = linear_7.view(32, -1, 12, 64);  linear_7 = None
        tmp_85 = tmp_84.transpose(1, 2);  tmp_84 = None
        linear_8 = torch.nn.functional.linear(tmp_79, in_33, in_32);  in_33 = in_32 = None
        tmp_87 = linear_8.view(32, -1, 12, 64);  linear_8 = None
        tmp_88 = tmp_87.transpose(1, 2);  tmp_87 = None
        to_16 = tmp_57.to(torch.float16);  tmp_57 = None
        scaled_dot_product_attention_1 = torch.nn.functional.scaled_dot_product_attention(tmp_82, tmp_85, tmp_88, attn_mask = to_16, dropout_p = 0.0, is_causal = False);  tmp_82 = tmp_85 = tmp_88 = to_16 = None
        tmp_90 = scaled_dot_product_attention_1.transpose(1, 2);  scaled_dot_product_attention_1 = None
        tmp_91 = tmp_90.reshape(32, 64, 768);  tmp_90 = None
        linear_9 = torch.nn.functional.linear(tmp_91, in_27, in_26);  tmp_91 = in_27 = in_26 = None
        tmp_93 = torch.nn.functional.dropout(linear_9, 0.1, False, False);  linear_9 = None
        tmp_94 = tmp_93 + tmp_79;  tmp_93 = tmp_79 = None
        tmp_95 = torch.nn.functional.layer_norm(tmp_94, (768,), in_25, in_24, 1e-12);  tmp_94 = in_25 = in_24 = None
        linear_10 = torch.nn.functional.linear(tmp_95, in_35, in_34);  in_35 = in_34 = None
        tmp_97 = torch.nn.functional.gelu(linear_10);  linear_10 = None
        linear_11 = torch.nn.functional.linear(tmp_97, in_39, in_38);  tmp_97 = in_39 = in_38 = None
        tmp_99 = torch.nn.functional.dropout(linear_11, 0.1, False, False);  linear_11 = None
        tmp_100 = tmp_99 + tmp_95;  tmp_99 = tmp_95 = None
        tmp_101 = torch.nn.functional.layer_norm(tmp_100, (768,), in_37, in_36, 1e-12);  tmp_100 = in_37 = in_36 = None
        tmp_102 = tmp_101[(slice(None, None, None), 0)]
        linear_12 = torch.nn.functional.linear(tmp_102, in_41, in_40);  tmp_102 = in_41 = in_40 = None
        tmp_104 = torch.tanh(linear_12);  linear_12 = None
        return (tmp_101, tmp_104)
        
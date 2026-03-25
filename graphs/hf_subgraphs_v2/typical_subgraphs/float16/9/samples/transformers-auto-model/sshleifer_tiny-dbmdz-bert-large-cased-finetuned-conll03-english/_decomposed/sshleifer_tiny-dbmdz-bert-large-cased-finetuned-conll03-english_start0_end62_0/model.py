import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, w_28 : torch.Tensor, w_29 : torch.Tensor, w_30 : torch.Tensor, w_31 : torch.Tensor, w_32 : torch.Tensor, w_33 : torch.Tensor, w_34 : torch.Tensor, w_35 : torch.Tensor, w_36 : torch.Tensor, w_37 : torch.Tensor, w_38 : torch.Tensor, w_39 : torch.Tensor, in_2 : torch.Tensor):
        tmp_43 = w_0[(slice(None, None, None), slice(0, 13, None))];  w_0 = None
        tmp_44 = torch.nn.functional.embedding(in_1, w_5, 0, None, 2.0, False, False);  in_1 = w_5 = None
        tmp_45 = torch.nn.functional.embedding(in_2, w_4, None, None, 2.0, False, False);  in_2 = w_4 = None
        tmp_46 = tmp_44 + tmp_45;  tmp_44 = tmp_45 = None
        tmp_47 = torch.nn.functional.embedding(tmp_43, w_3, None, None, 2.0, False, False);  tmp_43 = w_3 = None
        tmp_46 += tmp_47;  tmp_48 = tmp_46;  tmp_46 = tmp_47 = None
        tmp_49 = torch.nn.functional.layer_norm(tmp_48, (2,), w_2, w_1, 1e-12);  tmp_48 = w_2 = w_1 = None
        tmp_50 = torch.nn.functional.dropout(tmp_49, 0.1, False, False);  tmp_49 = None
        tmp_51 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_52 = tmp_51.expand(1, 1, 13, 13);  tmp_51 = None
        tmp_53 = tmp_52.to(torch.float32);  tmp_52 = None
        tmp_54 = torch.tensor(1.0, dtype = torch.float32)
        tmp_55 = tmp_54 - tmp_53;  tmp_54 = tmp_53 = None
        tmp_56 = tmp_55.to(torch.bool)
        tmp_57 = tmp_55.masked_fill(tmp_56, -3.4028234663852886e+38);  tmp_55 = tmp_56 = None
        linear = torch.nn.functional.linear(tmp_50, w_13, w_12);  w_13 = w_12 = None
        tmp_59 = linear.view(1, -1, 2, 1);  linear = None
        tmp_60 = tmp_59.transpose(1, 2);  tmp_59 = None
        linear_1 = torch.nn.functional.linear(tmp_50, w_11, w_10);  w_11 = w_10 = None
        tmp_62 = linear_1.view(1, -1, 2, 1);  linear_1 = None
        tmp_63 = tmp_62.transpose(1, 2);  tmp_62 = None
        linear_2 = torch.nn.functional.linear(tmp_50, w_15, w_14);  w_15 = w_14 = None
        tmp_65 = linear_2.view(1, -1, 2, 1);  linear_2 = None
        tmp_66 = tmp_65.transpose(1, 2);  tmp_65 = None
        to_6 = tmp_57.to(torch.float16)
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_60, tmp_63, tmp_66, attn_mask = to_6, dropout_p = 0.0, is_causal = False);  tmp_60 = tmp_63 = tmp_66 = to_6 = None
        tmp_68 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_69 = tmp_68.reshape(1, 13, 2);  tmp_68 = None
        linear_3 = torch.nn.functional.linear(tmp_69, w_9, w_8);  tmp_69 = w_9 = w_8 = None
        tmp_71 = torch.nn.functional.dropout(linear_3, 0.1, False, False);  linear_3 = None
        tmp_72 = tmp_71 + tmp_50;  tmp_71 = tmp_50 = None
        tmp_73 = torch.nn.functional.layer_norm(tmp_72, (2,), w_7, w_6, 1e-12);  tmp_72 = w_7 = w_6 = None
        linear_4 = torch.nn.functional.linear(tmp_73, w_17, w_16);  w_17 = w_16 = None
        tmp_75 = torch.nn.functional.gelu(linear_4);  linear_4 = None
        linear_5 = torch.nn.functional.linear(tmp_75, w_21, w_20);  tmp_75 = w_21 = w_20 = None
        tmp_77 = torch.nn.functional.dropout(linear_5, 0.1, False, False);  linear_5 = None
        tmp_78 = tmp_77 + tmp_73;  tmp_77 = tmp_73 = None
        tmp_79 = torch.nn.functional.layer_norm(tmp_78, (2,), w_19, w_18, 1e-12);  tmp_78 = w_19 = w_18 = None
        linear_6 = torch.nn.functional.linear(tmp_79, w_29, w_28);  w_29 = w_28 = None
        tmp_81 = linear_6.view(1, -1, 2, 1);  linear_6 = None
        tmp_82 = tmp_81.transpose(1, 2);  tmp_81 = None
        linear_7 = torch.nn.functional.linear(tmp_79, w_27, w_26);  w_27 = w_26 = None
        tmp_84 = linear_7.view(1, -1, 2, 1);  linear_7 = None
        tmp_85 = tmp_84.transpose(1, 2);  tmp_84 = None
        linear_8 = torch.nn.functional.linear(tmp_79, w_31, w_30);  w_31 = w_30 = None
        tmp_87 = linear_8.view(1, -1, 2, 1);  linear_8 = None
        tmp_88 = tmp_87.transpose(1, 2);  tmp_87 = None
        to_16 = tmp_57.to(torch.float16);  tmp_57 = None
        scaled_dot_product_attention_1 = torch.nn.functional.scaled_dot_product_attention(tmp_82, tmp_85, tmp_88, attn_mask = to_16, dropout_p = 0.0, is_causal = False);  tmp_82 = tmp_85 = tmp_88 = to_16 = None
        tmp_90 = scaled_dot_product_attention_1.transpose(1, 2);  scaled_dot_product_attention_1 = None
        tmp_91 = tmp_90.reshape(1, 13, 2);  tmp_90 = None
        linear_9 = torch.nn.functional.linear(tmp_91, w_25, w_24);  tmp_91 = w_25 = w_24 = None
        tmp_93 = torch.nn.functional.dropout(linear_9, 0.1, False, False);  linear_9 = None
        tmp_94 = tmp_93 + tmp_79;  tmp_93 = tmp_79 = None
        tmp_95 = torch.nn.functional.layer_norm(tmp_94, (2,), w_23, w_22, 1e-12);  tmp_94 = w_23 = w_22 = None
        linear_10 = torch.nn.functional.linear(tmp_95, w_33, w_32);  w_33 = w_32 = None
        tmp_97 = torch.nn.functional.gelu(linear_10);  linear_10 = None
        linear_11 = torch.nn.functional.linear(tmp_97, w_37, w_36);  tmp_97 = w_37 = w_36 = None
        tmp_99 = torch.nn.functional.dropout(linear_11, 0.1, False, False);  linear_11 = None
        tmp_100 = tmp_99 + tmp_95;  tmp_99 = tmp_95 = None
        tmp_101 = torch.nn.functional.layer_norm(tmp_100, (2,), w_35, w_34, 1e-12);  tmp_100 = w_35 = w_34 = None
        tmp_102 = tmp_101[(slice(None, None, None), 0)]
        linear_12 = torch.nn.functional.linear(tmp_102, w_39, w_38);  tmp_102 = w_39 = w_38 = None
        tmp_104 = torch.tanh(linear_12);  linear_12 = None
        return (tmp_101, tmp_104)
        
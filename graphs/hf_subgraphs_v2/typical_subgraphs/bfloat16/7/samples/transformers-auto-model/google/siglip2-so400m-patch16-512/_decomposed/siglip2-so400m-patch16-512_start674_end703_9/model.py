import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor, in_29 : torch.Tensor, in_30 : torch.Tensor, in_31 : torch.Tensor, in_32 : torch.Tensor, in_33 : torch.Tensor, in_34 : torch.Tensor):
        linear = torch.nn.functional.linear(in_33, in_19, in_18);  in_33 = in_19 = in_18 = None
        tmp_34 = in_34 + linear;  in_34 = linear = None
        tmp_35 = torch.nn.functional.layer_norm(tmp_34, (1152,), in_13, in_12, 1e-06);  in_13 = in_12 = None
        to = tmp_35.to(torch.bfloat16);  tmp_35 = None
        linear_1 = torch.nn.functional.linear(to, in_15, in_14);  to = in_15 = in_14 = None
        tmp_37 = torch.nn.functional.gelu(linear_1, approximate = 'tanh');  linear_1 = None
        to_1 = tmp_37.to(torch.bfloat16);  tmp_37 = None
        linear_2 = torch.nn.functional.linear(to_1, in_17, in_16);  to_1 = in_17 = in_16 = None
        tmp_39 = tmp_34 + linear_2;  tmp_34 = linear_2 = None
        tmp_40 = torch.nn.functional.layer_norm(tmp_39, (1152,), in_32, in_31, 1e-06);  tmp_39 = in_32 = in_31 = None
        tmp_41 = in_30.repeat(1, 1, 1);  in_30 = None
        tmp_42 = tmp_41.transpose(1, 0);  tmp_41 = None
        tmp_43 = tmp_40.transpose(1, 0)
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(tmp_42, tmp_43, tmp_43, 1152, 16, in_23, in_22, None, None, False, 0.0, in_21, in_20, training = False, key_padding_mask = None, need_weights = True, attn_mask = None, average_attn_weights = True, is_causal = False);  tmp_42 = tmp_43 = in_23 = in_22 = in_21 = in_20 = None
        tmp_45 = multi_head_attention_forward[0];  multi_head_attention_forward = None
        tmp_46 = tmp_45.transpose(1, 0);  tmp_45 = None
        tmp_47 = torch.nn.functional.layer_norm(tmp_46, (1152,), in_25, in_24, 1e-06);  in_25 = in_24 = None
        to_2 = tmp_47.to(torch.bfloat16);  tmp_47 = None
        linear_3 = torch.nn.functional.linear(to_2, in_27, in_26);  to_2 = in_27 = in_26 = None
        tmp_49 = torch.nn.functional.gelu(linear_3, approximate = 'tanh');  linear_3 = None
        to_3 = tmp_49.to(torch.bfloat16);  tmp_49 = None
        linear_4 = torch.nn.functional.linear(to_3, in_29, in_28);  to_3 = in_29 = in_28 = None
        tmp_51 = tmp_46 + linear_4;  tmp_46 = linear_4 = None
        tmp_52 = tmp_51[(slice(None, None, None), 0)];  tmp_51 = None
        tmp_53 = in_0.view(-1, 7);  in_0 = None
        tmp_54 = in_1[(slice(None, None, None), slice(None, 7, None))];  in_1 = None
        tmp_55 = torch.nn.functional.embedding(tmp_53, in_3, None, None, 2.0, False, False);  tmp_53 = in_3 = None
        tmp_56 = torch.nn.functional.embedding(tmp_54, in_2, None, None, 2.0, False, False);  tmp_54 = in_2 = None
        tmp_57 = tmp_55 + tmp_56;  tmp_55 = tmp_56 = None
        tmp_58 = torch.nn.functional.layer_norm(tmp_57, (1152,), in_5, in_4, 1e-06);  in_5 = in_4 = None
        to_4 = tmp_58.to(torch.bfloat16)
        linear_5 = torch.nn.functional.linear(to_4, in_9, in_8);  to_4 = in_9 = in_8 = None
        to_5 = tmp_58.to(torch.bfloat16)
        linear_6 = torch.nn.functional.linear(to_5, in_7, in_6);  to_5 = in_7 = in_6 = None
        to_6 = tmp_58.to(torch.bfloat16);  tmp_58 = None
        linear_7 = torch.nn.functional.linear(to_6, in_11, in_10);  to_6 = in_11 = in_10 = None
        return (tmp_57, linear_6, tmp_40, tmp_52, linear_5, linear_7)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, w_21 : torch.Tensor, w_22 : torch.Tensor, w_23 : torch.Tensor, w_24 : torch.Tensor, w_25 : torch.Tensor, w_26 : torch.Tensor, w_27 : torch.Tensor, w_28 : torch.Tensor, w_29 : torch.Tensor, w_30 : torch.Tensor, w_31 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_1, w_18, w_17);  in_1 = w_18 = w_17 = None
        tmp_34 = in_2 + linear;  in_2 = linear = None
        tmp_35 = torch.nn.functional.layer_norm(tmp_34, (1152,), w_12, w_11, 1e-06);  w_12 = w_11 = None
        linear_1 = torch.nn.functional.linear(tmp_35, w_14, w_13);  tmp_35 = w_14 = w_13 = None
        tmp_37 = torch.nn.functional.gelu(linear_1, approximate = 'tanh');  linear_1 = None
        linear_2 = torch.nn.functional.linear(tmp_37, w_16, w_15);  tmp_37 = w_16 = w_15 = None
        tmp_39 = tmp_34 + linear_2;  tmp_34 = linear_2 = None
        tmp_40 = torch.nn.functional.layer_norm(tmp_39, (1152,), w_31, w_30, 1e-06);  tmp_39 = w_31 = w_30 = None
        tmp_41 = w_29.repeat(1, 1, 1);  w_29 = None
        tmp_42 = tmp_41.transpose(1, 0);  tmp_41 = None
        tmp_43 = tmp_40.transpose(1, 0)
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(tmp_42, tmp_43, tmp_43, 1152, 16, w_22, w_21, None, None, False, 0.0, w_20, w_19, training = False, key_padding_mask = None, need_weights = True, attn_mask = None, average_attn_weights = True, is_causal = False);  tmp_42 = tmp_43 = w_22 = w_21 = w_20 = w_19 = None
        tmp_45 = multi_head_attention_forward[0];  multi_head_attention_forward = None
        tmp_46 = tmp_45.transpose(1, 0);  tmp_45 = None
        tmp_47 = torch.nn.functional.layer_norm(tmp_46, (1152,), w_24, w_23, 1e-06);  w_24 = w_23 = None
        linear_3 = torch.nn.functional.linear(tmp_47, w_26, w_25);  tmp_47 = w_26 = w_25 = None
        tmp_49 = torch.nn.functional.gelu(linear_3, approximate = 'tanh');  linear_3 = None
        linear_4 = torch.nn.functional.linear(tmp_49, w_28, w_27);  tmp_49 = w_28 = w_27 = None
        tmp_51 = tmp_46 + linear_4;  tmp_46 = linear_4 = None
        tmp_52 = tmp_51[(slice(None, None, None), 0)];  tmp_51 = None
        tmp_53 = in_0.view(-1, 7);  in_0 = None
        tmp_54 = w_0[(slice(None, None, None), slice(None, 7, None))];  w_0 = None
        tmp_55 = torch.nn.functional.embedding(tmp_53, w_2, None, None, 2.0, False, False);  tmp_53 = w_2 = None
        tmp_56 = torch.nn.functional.embedding(tmp_54, w_1, None, None, 2.0, False, False);  tmp_54 = w_1 = None
        tmp_57 = tmp_55 + tmp_56;  tmp_55 = tmp_56 = None
        tmp_58 = torch.nn.functional.layer_norm(tmp_57, (1152,), w_4, w_3, 1e-06);  w_4 = w_3 = None
        linear_5 = torch.nn.functional.linear(tmp_58, w_8, w_7);  w_8 = w_7 = None
        linear_6 = torch.nn.functional.linear(tmp_58, w_6, w_5);  w_6 = w_5 = None
        linear_7 = torch.nn.functional.linear(tmp_58, w_10, w_9);  tmp_58 = w_10 = w_9 = None
        return (tmp_57, linear_6, tmp_40, tmp_52, linear_5, linear_7)
        
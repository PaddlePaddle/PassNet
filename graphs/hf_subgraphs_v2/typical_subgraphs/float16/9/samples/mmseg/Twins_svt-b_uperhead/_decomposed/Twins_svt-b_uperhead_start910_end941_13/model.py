import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_18 = torch.nn.functional.gelu(in_0, approximate = 'none');  in_0 = None
        tmp_19 = torch.nn.functional.dropout(tmp_18, 0.0, False, False);  tmp_18 = None
        linear = torch.nn.functional.linear(tmp_19, w_1, w_0);  tmp_19 = w_1 = w_0 = None
        tmp_21 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_22 = in_1 + tmp_21;  in_1 = tmp_21 = None
        tmp_23 = tmp_22.transpose(1, 2);  tmp_22 = None
        tmp_24 = tmp_23.view(1, 768, 16, 16);  tmp_23 = None
        conv2d = torch.conv2d(tmp_24, w_17, w_16, (1, 1), (1, 1), (1, 1), 768);  w_17 = w_16 = None
        tmp_26 = conv2d + tmp_24;  conv2d = tmp_24 = None
        tmp_27 = tmp_26.flatten(2);  tmp_26 = None
        tmp_28 = tmp_27.transpose(1, 2);  tmp_27 = None
        tmp_29 = torch.nn.functional.layer_norm(tmp_28, (768,), w_11, w_10, 1e-05);  w_11 = w_10 = None
        tmp_30 = tmp_29.transpose(0, 1)
        tmp_31 = tmp_29.transpose(0, 1);  tmp_29 = None
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(tmp_30, tmp_31, tmp_31, 768, 24, w_5, w_4, None, None, False, 0.0, w_3, w_2, training = False, key_padding_mask = None, need_weights = True, attn_mask = None, average_attn_weights = True, is_causal = False);  tmp_30 = tmp_31 = w_5 = w_4 = w_3 = w_2 = None
        tmp_33 = multi_head_attention_forward[0];  multi_head_attention_forward = None
        tmp_34 = tmp_33.transpose(0, 1);  tmp_33 = None
        tmp_35 = torch.nn.functional.dropout(tmp_34, 0.0, False, False);  tmp_34 = None
        tmp_36 = 0.0 + tmp_35;  tmp_35 = None
        tmp_37 = tmp_28 + tmp_36;  tmp_28 = tmp_36 = None
        tmp_38 = torch.nn.functional.layer_norm(tmp_37, (768,), w_13, w_12, 1e-05);  w_13 = w_12 = None
        linear_1 = torch.nn.functional.linear(tmp_38, w_7, w_6);  tmp_38 = w_7 = w_6 = None
        tmp_40 = torch.nn.functional.gelu(linear_1, approximate = 'none');  linear_1 = None
        tmp_41 = torch.nn.functional.dropout(tmp_40, 0.0, False, False);  tmp_40 = None
        linear_2 = torch.nn.functional.linear(tmp_41, w_9, w_8);  tmp_41 = w_9 = w_8 = None
        tmp_43 = torch.nn.functional.dropout(linear_2, 0.0, False, False);  linear_2 = None
        tmp_44 = tmp_37 + tmp_43;  tmp_37 = tmp_43 = None
        tmp_45 = torch.nn.functional.layer_norm(tmp_44, (768,), w_15, w_14, 1e-05);  tmp_44 = w_15 = w_14 = None
        tmp_46 = tmp_45.reshape(1, 16, 16, -1);  tmp_45 = None
        tmp_47 = tmp_46.permute(0, 3, 1, 2);  tmp_46 = None
        tmp_48 = tmp_47.contiguous();  tmp_47 = None
        return (tmp_48,)
        
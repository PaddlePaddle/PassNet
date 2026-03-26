import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_16 = torch.nn.functional.gelu(in_0, approximate = 'none');  in_0 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.0, False, False);  tmp_16 = None
        linear = torch.nn.functional.linear(tmp_17, w_1, w_0);  tmp_17 = w_1 = w_0 = None
        tmp_19 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_20 = in_1 + tmp_19;  in_1 = tmp_19 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (512,), w_11, w_10, 1e-05);  w_11 = w_10 = None
        tmp_22 = tmp_21.transpose(0, 1)
        tmp_23 = tmp_21.transpose(0, 1);  tmp_21 = None
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(tmp_22, tmp_23, tmp_23, 512, 16, w_5, w_4, None, None, False, 0.0, w_3, w_2, training = False, key_padding_mask = None, need_weights = True, attn_mask = None, average_attn_weights = True, is_causal = False);  tmp_22 = tmp_23 = w_5 = w_4 = w_3 = w_2 = None
        tmp_25 = multi_head_attention_forward[0];  multi_head_attention_forward = None
        tmp_26 = tmp_25.transpose(0, 1);  tmp_25 = None
        tmp_27 = torch.nn.functional.dropout(tmp_26, 0.0, False, False);  tmp_26 = None
        tmp_28 = 0.0 + tmp_27;  tmp_27 = None
        tmp_29 = tmp_20 + tmp_28;  tmp_20 = tmp_28 = None
        tmp_30 = torch.nn.functional.layer_norm(tmp_29, (512,), w_13, w_12, 1e-05);  w_13 = w_12 = None
        linear_1 = torch.nn.functional.linear(tmp_30, w_7, w_6);  tmp_30 = w_7 = w_6 = None
        tmp_32 = torch.nn.functional.gelu(linear_1, approximate = 'none');  linear_1 = None
        tmp_33 = torch.nn.functional.dropout(tmp_32, 0.0, False, False);  tmp_32 = None
        linear_2 = torch.nn.functional.linear(tmp_33, w_9, w_8);  tmp_33 = w_9 = w_8 = None
        tmp_35 = torch.nn.functional.dropout(linear_2, 0.0, False, False);  linear_2 = None
        tmp_36 = tmp_29 + tmp_35;  tmp_29 = tmp_35 = None
        tmp_37 = torch.nn.functional.layer_norm(tmp_36, (512,), w_15, w_14, 1e-05);  tmp_36 = w_15 = w_14 = None
        tmp_38 = tmp_37.reshape(1, 16, 16, -1);  tmp_37 = None
        tmp_39 = tmp_38.permute(0, 3, 1, 2);  tmp_38 = None
        tmp_40 = tmp_39.contiguous();  tmp_39 = None
        return (tmp_40,)
        
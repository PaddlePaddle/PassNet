import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, in_0, in_1, in_2):
        tmp_9 = torch.nn.functional.gelu(in_2);  in_2 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.0, False, False);  tmp_9 = None
        linear = torch.nn.functional.linear(tmp_10, w_1, w_0);  tmp_10 = w_1 = w_0 = None
        tmp_12 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_13 = in_1 + tmp_12;  in_1 = tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (768,), w_3, w_2, 1e-05);  tmp_13 = w_3 = w_2 = None
        tmp_15 = tmp_14.view(1, 199, 12, -1)
        tmp_16 = tmp_15.permute(0, 2, 1, 3);  tmp_15 = None
        linear_1 = torch.nn.functional.linear(tmp_16, w_7, w_6);  tmp_16 = w_7 = w_6 = None
        tmp_18 = linear_1.view(1, 12, 199, 2, 4);  linear_1 = None
        tmp_19 = tmp_18.sum(-1, keepdim = False);  tmp_18 = None
        tmp_20 = torch.sigmoid(tmp_19);  tmp_19 = None
        chunk = tmp_20.chunk(2, dim = -1);  tmp_20 = None
        tmp_22 = chunk[0]
        tmp_23 = chunk[1];  chunk = None
        tmp_24 = tmp_23 * w_8;  tmp_23 = w_8 = None
        tmp_25 = tmp_24 - 1.0;  tmp_24 = None
        tmp_26 = tmp_22 * tmp_25;  tmp_22 = tmp_25 = None
        tmp_27 = tmp_26 + 2.0;  tmp_26 = None
        tmp_28 = tmp_27.view(1, 12, -1, 1);  tmp_27 = None
        tmp_29 = tmp_28 * in_0;  tmp_28 = in_0 = None
        tmp_30 = tmp_29.view((1, 12, 199, 199));  tmp_29 = None
        linear_2 = torch.nn.functional.linear(tmp_14, w_5, w_4);  w_5 = w_4 = None
        chunk_1 = linear_2.chunk(3, -1);  linear_2 = None
        tmp_33 = chunk_1[0]
        tmp_34 = chunk_1[1]
        tmp_35 = chunk_1[2];  chunk_1 = None
        tmp_36 = tmp_33.view((1, 199, 12, 64));  tmp_33 = None
        tmp_37 = tmp_36.transpose(2, 1);  tmp_36 = None
        tmp_38 = tmp_34.view((1, 199, 12, 64));  tmp_34 = None
        tmp_39 = tmp_38.transpose(2, 1);  tmp_38 = None
        tmp_40 = tmp_35.view((1, 199, 12, 64));  tmp_35 = None
        tmp_41 = tmp_40.transpose(2, 1);  tmp_40 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_37, tmp_39, tmp_41, attn_mask = tmp_30, dropout_p = 0.0, is_causal = False);  tmp_37 = tmp_39 = tmp_41 = tmp_30 = None
        tmp_43 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_44 = tmp_43.reshape(1, -1, 768);  tmp_43 = None
        return (tmp_44, tmp_14)
        
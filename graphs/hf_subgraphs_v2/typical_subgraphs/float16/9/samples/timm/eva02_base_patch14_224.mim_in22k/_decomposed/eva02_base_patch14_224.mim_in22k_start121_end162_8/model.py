import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_4 = in_0.reshape(1, 257, 12, -1);  in_0 = None
        tmp_5 = tmp_4.transpose(1, 2);  tmp_4 = None
        linear = torch.nn.functional.linear(in_1, w_0, None);  w_0 = None
        tmp_7 = linear.reshape(1, 257, 12, -1);  linear = None
        tmp_8 = tmp_7.transpose(1, 2);  tmp_7 = None
        linear_1 = torch.nn.functional.linear(in_1, w_2, w_1);  in_1 = w_2 = w_1 = None
        tmp_10 = linear_1.reshape(1, 257, 12, -1);  linear_1 = None
        tmp_11 = tmp_10.transpose(1, 2);  tmp_10 = None
        tmp_12 = tmp_5[(slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None))]
        tmp_13 = tmp_5[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  tmp_5 = None
        tensor_split = w_3.tensor_split(2, -1)
        tmp_15 = tensor_split[0]
        tmp_16 = tensor_split[1];  tensor_split = None
        tmp_17 = tmp_13 * tmp_16;  tmp_16 = None
        tmp_18 = tmp_13[(Ellipsis, slice(1, None, 2))]
        tmp_19 = -tmp_18;  tmp_18 = None
        tmp_20 = tmp_13[(Ellipsis, slice(None, None, 2))];  tmp_13 = None
        tmp_21 = torch.stack([tmp_19, tmp_20], -1);  tmp_19 = tmp_20 = None
        tmp_22 = tmp_21.reshape((1, 12, 256, 64));  tmp_21 = None
        tmp_23 = tmp_22 * tmp_15;  tmp_22 = tmp_15 = None
        tmp_24 = tmp_17 + tmp_23;  tmp_17 = tmp_23 = None
        tmp_25 = torch.cat([tmp_12, tmp_24], dim = 2);  tmp_12 = tmp_24 = None
        tmp_26 = tmp_25.type_as(tmp_11);  tmp_25 = None
        tmp_27 = tmp_8[(slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None))]
        tmp_28 = tmp_8[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  tmp_8 = None
        tensor_split_1 = w_3.tensor_split(2, -1);  w_3 = None
        tmp_30 = tensor_split_1[0]
        tmp_31 = tensor_split_1[1];  tensor_split_1 = None
        tmp_32 = tmp_28 * tmp_31;  tmp_31 = None
        tmp_33 = tmp_28[(Ellipsis, slice(1, None, 2))]
        tmp_34 = -tmp_33;  tmp_33 = None
        tmp_35 = tmp_28[(Ellipsis, slice(None, None, 2))];  tmp_28 = None
        tmp_36 = torch.stack([tmp_34, tmp_35], -1);  tmp_34 = tmp_35 = None
        tmp_37 = tmp_36.reshape((1, 12, 256, 64));  tmp_36 = None
        tmp_38 = tmp_37 * tmp_30;  tmp_37 = tmp_30 = None
        tmp_39 = tmp_32 + tmp_38;  tmp_32 = tmp_38 = None
        tmp_40 = torch.cat([tmp_27, tmp_39], dim = 2);  tmp_27 = tmp_39 = None
        tmp_41 = tmp_40.type_as(tmp_11);  tmp_40 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_26, tmp_41, tmp_11, attn_mask = None, dropout_p = 0.0);  tmp_26 = tmp_41 = tmp_11 = None
        tmp_43 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_44 = tmp_43.reshape(1, 257, 768);  tmp_43 = None
        return (tmp_44,)
        
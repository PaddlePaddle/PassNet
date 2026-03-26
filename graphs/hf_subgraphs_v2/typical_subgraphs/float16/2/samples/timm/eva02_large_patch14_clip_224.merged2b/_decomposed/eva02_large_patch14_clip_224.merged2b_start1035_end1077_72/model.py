import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor):
        tmp_6 = in_6.reshape(1, 257, 16, -1);  in_6 = None
        tmp_7 = tmp_6.transpose(1, 2);  tmp_6 = None
        linear = torch.nn.functional.linear(in_7, in_0, None);  in_0 = None
        tmp_9 = linear.reshape(1, 257, 16, -1);  linear = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        linear_1 = torch.nn.functional.linear(in_7, in_4, in_3);  in_7 = in_4 = in_3 = None
        tmp_12 = linear_1.reshape(1, 257, 16, -1);  linear_1 = None
        tmp_13 = tmp_12.transpose(1, 2);  tmp_12 = None
        tmp_14 = tmp_7[(slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None))]
        tmp_15 = tmp_7[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  tmp_7 = None
        tensor_split = in_5.tensor_split(2, -1)
        tmp_17 = tensor_split[0]
        tmp_18 = tensor_split[1];  tensor_split = None
        tmp_19 = tmp_15 * tmp_18;  tmp_18 = None
        tmp_20 = tmp_15[(Ellipsis, slice(1, None, 2))]
        tmp_21 = -tmp_20;  tmp_20 = None
        tmp_22 = tmp_15[(Ellipsis, slice(None, None, 2))];  tmp_15 = None
        tmp_23 = torch.stack([tmp_21, tmp_22], -1);  tmp_21 = tmp_22 = None
        tmp_24 = tmp_23.reshape((1, 16, 256, 64));  tmp_23 = None
        tmp_25 = tmp_24 * tmp_17;  tmp_24 = tmp_17 = None
        tmp_26 = tmp_19 + tmp_25;  tmp_19 = tmp_25 = None
        tmp_27 = torch.cat([tmp_14, tmp_26], dim = 2);  tmp_14 = tmp_26 = None
        tmp_28 = tmp_27.type_as(tmp_13);  tmp_27 = None
        tmp_29 = tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None))]
        tmp_30 = tmp_10[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  tmp_10 = None
        tensor_split_1 = in_5.tensor_split(2, -1);  in_5 = None
        tmp_32 = tensor_split_1[0]
        tmp_33 = tensor_split_1[1];  tensor_split_1 = None
        tmp_34 = tmp_30 * tmp_33;  tmp_33 = None
        tmp_35 = tmp_30[(Ellipsis, slice(1, None, 2))]
        tmp_36 = -tmp_35;  tmp_35 = None
        tmp_37 = tmp_30[(Ellipsis, slice(None, None, 2))];  tmp_30 = None
        tmp_38 = torch.stack([tmp_36, tmp_37], -1);  tmp_36 = tmp_37 = None
        tmp_39 = tmp_38.reshape((1, 16, 256, 64));  tmp_38 = None
        tmp_40 = tmp_39 * tmp_32;  tmp_39 = tmp_32 = None
        tmp_41 = tmp_34 + tmp_40;  tmp_34 = tmp_40 = None
        tmp_42 = torch.cat([tmp_29, tmp_41], dim = 2);  tmp_29 = tmp_41 = None
        tmp_43 = tmp_42.type_as(tmp_13);  tmp_42 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_28, tmp_43, tmp_13, attn_mask = None, dropout_p = 0.0);  tmp_28 = tmp_43 = tmp_13 = None
        tmp_45 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_46 = tmp_45.reshape(1, 257, 1024);  tmp_45 = None
        tmp_47 = torch.nn.functional.layer_norm(tmp_46, (1024,), in_2, in_1, 1e-06);  tmp_46 = in_2 = in_1 = None
        return (tmp_47,)
        
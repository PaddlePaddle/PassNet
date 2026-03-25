import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, in_0 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_8, w_7, (14, 14), (0, 0), (1, 1), 1);  in_0 = w_8 = w_7 = None
        tmp_14 = conv2d.flatten(2);  conv2d = None
        tmp_15 = tmp_14.transpose(1, 2);  tmp_14 = None
        tmp_16 = w_10.expand(1, -1, -1);  w_10 = None
        tmp_17 = torch.cat([tmp_16, tmp_15], dim = 1);  tmp_16 = tmp_15 = None
        tmp_18 = tmp_17 + w_11;  tmp_17 = w_11 = None
        tmp_19 = torch.nn.functional.dropout(tmp_18, 0.0, False, False);  tmp_18 = None
        tmp_20 = torch.nn.functional.layer_norm(tmp_19, (768,), w_6, w_5, 1e-06);  w_6 = w_5 = None
        linear = torch.nn.functional.linear(tmp_20, w_2, w_1);  w_2 = w_1 = None
        tmp_22 = linear.reshape(1, 257, 12, -1);  linear = None
        tmp_23 = tmp_22.transpose(1, 2);  tmp_22 = None
        linear_1 = torch.nn.functional.linear(tmp_20, w_0, None);  w_0 = None
        tmp_25 = linear_1.reshape(1, 257, 12, -1);  linear_1 = None
        tmp_26 = tmp_25.transpose(1, 2);  tmp_25 = None
        linear_2 = torch.nn.functional.linear(tmp_20, w_4, w_3);  tmp_20 = w_4 = w_3 = None
        tmp_28 = linear_2.reshape(1, 257, 12, -1);  linear_2 = None
        tmp_29 = tmp_28.transpose(1, 2);  tmp_28 = None
        tmp_30 = tmp_23[(slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None))]
        tmp_31 = tmp_23[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  tmp_23 = None
        tensor_split = w_9.tensor_split(2, -1)
        tmp_33 = tensor_split[0]
        tmp_34 = tensor_split[1];  tensor_split = None
        tmp_35 = tmp_31 * tmp_34;  tmp_34 = None
        tmp_36 = tmp_31[(Ellipsis, slice(1, None, 2))]
        tmp_37 = -tmp_36;  tmp_36 = None
        tmp_38 = tmp_31[(Ellipsis, slice(None, None, 2))];  tmp_31 = None
        tmp_39 = torch.stack([tmp_37, tmp_38], -1);  tmp_37 = tmp_38 = None
        tmp_40 = tmp_39.reshape((1, 12, 256, 64));  tmp_39 = None
        tmp_41 = tmp_40 * tmp_33;  tmp_40 = tmp_33 = None
        tmp_42 = tmp_35 + tmp_41;  tmp_35 = tmp_41 = None
        tmp_43 = torch.cat([tmp_30, tmp_42], dim = 2);  tmp_30 = tmp_42 = None
        tmp_44 = tmp_43.type_as(tmp_29);  tmp_43 = None
        tmp_45 = tmp_26[(slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None))]
        tmp_46 = tmp_26[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  tmp_26 = None
        tensor_split_1 = w_9.tensor_split(2, -1);  w_9 = None
        tmp_48 = tensor_split_1[0]
        tmp_49 = tensor_split_1[1];  tensor_split_1 = None
        tmp_50 = tmp_46 * tmp_49;  tmp_49 = None
        tmp_51 = tmp_46[(Ellipsis, slice(1, None, 2))]
        tmp_52 = -tmp_51;  tmp_51 = None
        tmp_53 = tmp_46[(Ellipsis, slice(None, None, 2))];  tmp_46 = None
        tmp_54 = torch.stack([tmp_52, tmp_53], -1);  tmp_52 = tmp_53 = None
        tmp_55 = tmp_54.reshape((1, 12, 256, 64));  tmp_54 = None
        tmp_56 = tmp_55 * tmp_48;  tmp_55 = tmp_48 = None
        tmp_57 = tmp_50 + tmp_56;  tmp_50 = tmp_56 = None
        tmp_58 = torch.cat([tmp_45, tmp_57], dim = 2);  tmp_45 = tmp_57 = None
        tmp_59 = tmp_58.type_as(tmp_29);  tmp_58 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_44, tmp_59, tmp_29, attn_mask = None, dropout_p = 0.0);  tmp_44 = tmp_59 = tmp_29 = None
        tmp_61 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_62 = tmp_61.reshape(1, 257, 768);  tmp_61 = None
        return (tmp_19, tmp_62)
        
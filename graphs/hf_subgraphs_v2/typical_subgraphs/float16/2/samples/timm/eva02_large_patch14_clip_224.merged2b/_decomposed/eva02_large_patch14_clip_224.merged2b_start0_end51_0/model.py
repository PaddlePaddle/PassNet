import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor):
        conv2d = torch.conv2d(in_14, in_10, in_9, (14, 14), (0, 0), (1, 1), 1);  in_14 = in_10 = in_9 = None
        tmp_16 = conv2d.flatten(2);  conv2d = None
        tmp_17 = tmp_16.transpose(1, 2);  tmp_16 = None
        tmp_18 = in_12.expand(1, -1, -1);  in_12 = None
        tmp_19 = torch.cat([tmp_18, tmp_17], dim = 1);  tmp_18 = tmp_17 = None
        tmp_20 = tmp_19 + in_13;  tmp_19 = in_13 = None
        tmp_21 = torch.nn.functional.dropout(tmp_20, 0.0, False, False);  tmp_20 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (1024,), in_8, in_7, 1e-06);  in_8 = in_7 = None
        linear = torch.nn.functional.linear(tmp_22, in_4, in_3);  in_4 = in_3 = None
        tmp_24 = linear.reshape(1, 257, 16, -1);  linear = None
        tmp_25 = tmp_24.transpose(1, 2);  tmp_24 = None
        linear_1 = torch.nn.functional.linear(tmp_22, in_0, None);  in_0 = None
        tmp_27 = linear_1.reshape(1, 257, 16, -1);  linear_1 = None
        tmp_28 = tmp_27.transpose(1, 2);  tmp_27 = None
        linear_2 = torch.nn.functional.linear(tmp_22, in_6, in_5);  tmp_22 = in_6 = in_5 = None
        tmp_30 = linear_2.reshape(1, 257, 16, -1);  linear_2 = None
        tmp_31 = tmp_30.transpose(1, 2);  tmp_30 = None
        tmp_32 = tmp_25[(slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None))]
        tmp_33 = tmp_25[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  tmp_25 = None
        tensor_split = in_11.tensor_split(2, -1)
        tmp_35 = tensor_split[0]
        tmp_36 = tensor_split[1];  tensor_split = None
        tmp_37 = tmp_33 * tmp_36;  tmp_36 = None
        tmp_38 = tmp_33[(Ellipsis, slice(1, None, 2))]
        tmp_39 = -tmp_38;  tmp_38 = None
        tmp_40 = tmp_33[(Ellipsis, slice(None, None, 2))];  tmp_33 = None
        tmp_41 = torch.stack([tmp_39, tmp_40], -1);  tmp_39 = tmp_40 = None
        tmp_42 = tmp_41.reshape((1, 16, 256, 64));  tmp_41 = None
        tmp_43 = tmp_42 * tmp_35;  tmp_42 = tmp_35 = None
        tmp_44 = tmp_37 + tmp_43;  tmp_37 = tmp_43 = None
        tmp_45 = torch.cat([tmp_32, tmp_44], dim = 2);  tmp_32 = tmp_44 = None
        tmp_46 = tmp_45.type_as(tmp_31);  tmp_45 = None
        tmp_47 = tmp_28[(slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None))]
        tmp_48 = tmp_28[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  tmp_28 = None
        tensor_split_1 = in_11.tensor_split(2, -1);  in_11 = None
        tmp_50 = tensor_split_1[0]
        tmp_51 = tensor_split_1[1];  tensor_split_1 = None
        tmp_52 = tmp_48 * tmp_51;  tmp_51 = None
        tmp_53 = tmp_48[(Ellipsis, slice(1, None, 2))]
        tmp_54 = -tmp_53;  tmp_53 = None
        tmp_55 = tmp_48[(Ellipsis, slice(None, None, 2))];  tmp_48 = None
        tmp_56 = torch.stack([tmp_54, tmp_55], -1);  tmp_54 = tmp_55 = None
        tmp_57 = tmp_56.reshape((1, 16, 256, 64));  tmp_56 = None
        tmp_58 = tmp_57 * tmp_50;  tmp_57 = tmp_50 = None
        tmp_59 = tmp_52 + tmp_58;  tmp_52 = tmp_58 = None
        tmp_60 = torch.cat([tmp_47, tmp_59], dim = 2);  tmp_47 = tmp_59 = None
        tmp_61 = tmp_60.type_as(tmp_31);  tmp_60 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_46, tmp_61, tmp_31, attn_mask = None, dropout_p = 0.0);  tmp_46 = tmp_61 = tmp_31 = None
        tmp_63 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_64 = tmp_63.reshape(1, 257, 1024);  tmp_63 = None
        tmp_65 = torch.nn.functional.layer_norm(tmp_64, (1024,), in_2, in_1, 1e-06);  tmp_64 = in_2 = in_1 = None
        return (tmp_21, tmp_65)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
        chunk = in_10.chunk(2, dim = -1);  in_10 = None
        tmp_10 = chunk[0]
        tmp_11 = chunk[1];  chunk = None
        tmp_12 = torch.nn.functional.silu(tmp_10, inplace = False);  tmp_10 = None
        tmp_13 = tmp_12 * tmp_11;  tmp_12 = tmp_11 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False);  tmp_13 = None
        linear = torch.nn.functional.linear(tmp_14, in_1, in_0);  tmp_14 = in_1 = in_0 = None
        tmp_16 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_17 = in_9 + tmp_16;  in_9 = tmp_16 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (192,), in_7, in_6, 1e-06);  in_7 = in_6 = None
        tmp_19 = torch.cat((in_4, in_2, in_5));  in_4 = in_2 = in_5 = None
        linear_1 = torch.nn.functional.linear(tmp_18, weight = in_3, bias = tmp_19);  tmp_18 = in_3 = tmp_19 = None
        tmp_21 = linear_1.reshape(1, 257, 3, 3, -1);  linear_1 = None
        tmp_22 = tmp_21.permute(2, 0, 3, 1, 4);  tmp_21 = None
        unbind = tmp_22.unbind(0);  tmp_22 = None
        tmp_24 = unbind[0]
        tmp_25 = unbind[1]
        tmp_26 = unbind[2];  unbind = None
        tmp_27 = tmp_24[(slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None))]
        tmp_28 = tmp_24[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  tmp_24 = None
        tensor_split = in_8.tensor_split(2, -1)
        tmp_30 = tensor_split[0]
        tmp_31 = tensor_split[1];  tensor_split = None
        tmp_32 = tmp_28 * tmp_31;  tmp_31 = None
        tmp_33 = tmp_28[(Ellipsis, slice(1, None, 2))]
        tmp_34 = -tmp_33;  tmp_33 = None
        tmp_35 = tmp_28[(Ellipsis, slice(None, None, 2))];  tmp_28 = None
        tmp_36 = torch.stack([tmp_34, tmp_35], -1);  tmp_34 = tmp_35 = None
        tmp_37 = tmp_36.reshape((1, 3, 256, 64));  tmp_36 = None
        tmp_38 = tmp_37 * tmp_30;  tmp_37 = tmp_30 = None
        tmp_39 = tmp_32 + tmp_38;  tmp_32 = tmp_38 = None
        tmp_40 = torch.cat([tmp_27, tmp_39], dim = 2);  tmp_27 = tmp_39 = None
        tmp_41 = tmp_40.type_as(tmp_26);  tmp_40 = None
        tmp_42 = tmp_25[(slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None))]
        tmp_43 = tmp_25[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  tmp_25 = None
        tensor_split_1 = in_8.tensor_split(2, -1);  in_8 = None
        tmp_45 = tensor_split_1[0]
        tmp_46 = tensor_split_1[1];  tensor_split_1 = None
        tmp_47 = tmp_43 * tmp_46;  tmp_46 = None
        tmp_48 = tmp_43[(Ellipsis, slice(1, None, 2))]
        tmp_49 = -tmp_48;  tmp_48 = None
        tmp_50 = tmp_43[(Ellipsis, slice(None, None, 2))];  tmp_43 = None
        tmp_51 = torch.stack([tmp_49, tmp_50], -1);  tmp_49 = tmp_50 = None
        tmp_52 = tmp_51.reshape((1, 3, 256, 64));  tmp_51 = None
        tmp_53 = tmp_52 * tmp_45;  tmp_52 = tmp_45 = None
        tmp_54 = tmp_47 + tmp_53;  tmp_47 = tmp_53 = None
        tmp_55 = torch.cat([tmp_42, tmp_54], dim = 2);  tmp_42 = tmp_54 = None
        tmp_56 = tmp_55.type_as(tmp_26);  tmp_55 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_41, tmp_56, tmp_26, attn_mask = None, dropout_p = 0.0);  tmp_41 = tmp_56 = tmp_26 = None
        tmp_58 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_59 = tmp_58.reshape(1, 257, 192);  tmp_58 = None
        return (tmp_17, tmp_59)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, in_0, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_1, w_2, w_1);  in_1 = w_2 = w_1 = None
        tmp_16 = linear.view(1, -1, 16, 64);  linear = None
        tmp_17 = tmp_16.transpose(1, 2);  tmp_16 = None
        tmp_18 = w_0[slice(None, 2209, None)]
        tmp_19 = tmp_18.reshape(1, 47, 47, -1);  tmp_18 = None
        tmp_20 = tmp_19.permute(0, 3, 1, 2);  tmp_19 = None
        tmp_21 = torch.nn.functional.interpolate(tmp_20, size = (47, 47), mode = 'bilinear');  tmp_20 = None
        tmp_22 = tmp_21.permute(0, 2, 3, 1);  tmp_21 = None
        tmp_23 = tmp_22.reshape(2209, -1);  tmp_22 = None
        tmp_24 = w_0[slice(2209, None, None)];  w_0 = None
        tmp_25 = torch.cat([tmp_23, tmp_24]);  tmp_23 = tmp_24 = None
        tmp_26 = torch.arange(24)
        tmp_27 = torch.arange(24)
        meshgrid = torch.functional.meshgrid(tmp_26, tmp_27, indexing = 'ij');  tmp_26 = tmp_27 = None
        tmp_29 = meshgrid[0]
        tmp_30 = meshgrid[1];  meshgrid = None
        tmp_31 = torch.stack((tmp_29, tmp_30));  tmp_29 = tmp_30 = None
        tmp_32 = torch.flatten(tmp_31, 1);  tmp_31 = None
        tmp_33 = tmp_32[(slice(None, None, None), slice(None, None, None), None)]
        tmp_34 = tmp_32[(slice(None, None, None), None, slice(None, None, None))];  tmp_32 = None
        tmp_35 = tmp_33 - tmp_34;  tmp_33 = tmp_34 = None
        tmp_36 = tmp_35.permute(1, 2, 0);  tmp_35 = None
        tmp_37 = tmp_36.contiguous();  tmp_36 = None
        tmp_38 = tmp_37[(slice(None, None, None), slice(None, None, None), 0)]
        tmp_38 += 23;  tmp_39 = tmp_38;  tmp_38 = None
        tmp_37[(slice(None, None, None), slice(None, None, None), 0)] = tmp_39;  setitem = tmp_37;  tmp_39 = setitem = None
        tmp_41 = tmp_37[(slice(None, None, None), slice(None, None, None), 1)]
        tmp_41 += 23;  tmp_42 = tmp_41;  tmp_41 = None
        tmp_37[(slice(None, None, None), slice(None, None, None), 1)] = tmp_42;  setitem_1 = tmp_37;  tmp_42 = setitem_1 = None
        tmp_44 = tmp_37[(slice(None, None, None), slice(None, None, None), 0)]
        tmp_44 *= 47;  tmp_45 = tmp_44;  tmp_44 = None
        tmp_37[(slice(None, None, None), slice(None, None, None), 0)] = tmp_45;  setitem_2 = tmp_37;  tmp_45 = setitem_2 = None
        tmp_47 = torch.zeros(size = (577, 577), dtype = torch.int64)
        tmp_48 = tmp_37.sum(-1);  tmp_37 = None
        tmp_47[(slice(1, None, None), slice(1, None, None))] = tmp_48;  setitem_3 = tmp_47;  tmp_48 = setitem_3 = None
        tmp_47[(0, slice(0, None, None))] = 2209;  setitem_4 = tmp_47;  setitem_4 = None
        tmp_47[(slice(0, None, None), 0)] = 2210;  setitem_5 = tmp_47;  setitem_5 = None
        tmp_47[(0, 0)] = 2211;  setitem_6 = tmp_47;  setitem_6 = None
        tmp_53 = tmp_47.view(-1);  tmp_47 = None
        tmp_54 = tmp_25[tmp_53];  tmp_25 = tmp_53 = None
        tmp_55 = tmp_54.view(577, 577, -1);  tmp_54 = None
        tmp_56 = tmp_55.permute(2, 0, 1);  tmp_55 = None
        tmp_57 = tmp_56.contiguous();  tmp_56 = None
        tmp_58 = tmp_57.unsqueeze(0);  tmp_57 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_3, in_0, tmp_17, attn_mask = tmp_58, dropout_p = 0.0, is_causal = False, scale = 0.125);  in_3 = in_0 = tmp_17 = tmp_58 = None
        tmp_60 = scaled_dot_product_attention.permute(0, 2, 1, 3);  scaled_dot_product_attention = None
        tmp_61 = tmp_60.contiguous();  tmp_60 = None
        tmp_62 = tmp_61.view(1, 577, 1024);  tmp_61 = None
        linear_1 = torch.nn.functional.linear(tmp_62, w_4, w_3);  tmp_62 = w_4 = w_3 = None
        tmp_64 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_65 = w_11 * tmp_64;  w_11 = tmp_64 = None
        tmp_66 = tmp_65 + in_2;  tmp_65 = in_2 = None
        tmp_67 = torch.nn.functional.layer_norm(tmp_66, (1024,), w_8, w_7, 1e-12);  w_8 = w_7 = None
        linear_2 = torch.nn.functional.linear(tmp_67, w_6, w_5);  tmp_67 = w_6 = w_5 = None
        tmp_69 = torch.nn.functional.gelu(linear_2);  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_69, w_10, w_9);  tmp_69 = w_10 = w_9 = None
        tmp_71 = torch.nn.functional.dropout(linear_3, 0.0, False, False);  linear_3 = None
        tmp_72 = w_12 * tmp_71;  w_12 = tmp_71 = None
        tmp_73 = tmp_72 + tmp_66;  tmp_72 = tmp_66 = None
        tmp_74 = torch.nn.functional.layer_norm(tmp_73, (1024,), w_14, w_13, 1e-12);  w_14 = w_13 = None
        return (tmp_74, tmp_73)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, in_0, in_1, in_2):
        tmp_11 = torch.nn.functional.gelu(in_0, approximate = 'none');  in_0 = None
        tmp_12 = tmp_11.flatten(2);  tmp_11 = None
        tmp_13 = tmp_12.transpose(1, 2);  tmp_12 = None
        tmp_14 = tmp_13.contiguous();  tmp_13 = None
        tmp_15 = in_2 + tmp_14;  in_2 = tmp_14 = None
        tmp_16 = tmp_15.permute(0, 2, 1);  tmp_15 = None
        tmp_17 = tmp_16.view(1, 78, 64, 48);  tmp_16 = None
        tmp_18 = in_1.view(1, 156, -1);  in_1 = None
        tmp_19 = tmp_18.permute(0, 2, 1);  tmp_18 = None
        tmp_20 = torch.nn.functional.layer_norm(tmp_19, (156,), w_8, w_7, 1e-06);  w_8 = w_7 = None
        tmp_21 = tmp_20.view(1, 32, 24, 156);  tmp_20 = None
        tmp_22 = torch.nn.functional.pad(tmp_21, (0, 0, 2, 2, 1, 2), 'constant', None);  tmp_21 = None
        tmp_23 = tmp_22.view(1, 5, 7, 4, 7, 156);  tmp_22 = None
        tmp_24 = tmp_23.permute(0, 1, 3, 2, 4, 5);  tmp_23 = None
        tmp_25 = tmp_24.reshape(-1, 49, 156);  tmp_24 = None
        linear = torch.nn.functional.linear(tmp_25, w_4, w_3);  tmp_25 = w_4 = w_3 = None
        tmp_27 = linear.reshape(20, 49, 3, 4, 39);  linear = None
        tmp_28 = tmp_27.permute(2, 0, 3, 1, 4);  tmp_27 = None
        tmp_29 = tmp_28[0]
        tmp_30 = tmp_28[1]
        tmp_31 = tmp_28[2];  tmp_28 = None
        item = w_6.item();  w_6 = None
        tmp_33 = tmp_29 * item;  tmp_29 = item = None
        tmp_34 = tmp_30.transpose(-2, -1);  tmp_30 = None
        matmul = tmp_33 @ tmp_34;  tmp_33 = tmp_34 = None
        tmp_36 = w_0.view(-1);  w_0 = None
        tmp_37 = w_5[tmp_36];  w_5 = tmp_36 = None
        tmp_38 = tmp_37.view(49, 49, -1);  tmp_37 = None
        tmp_39 = tmp_38.permute(2, 0, 1);  tmp_38 = None
        tmp_40 = tmp_39.contiguous();  tmp_39 = None
        tmp_41 = tmp_40.unsqueeze(0);  tmp_40 = None
        tmp_42 = matmul + tmp_41;  matmul = tmp_41 = None
        tmp_43 = torch.nn.functional.softmax(tmp_42, -1, _stacklevel = 5);  tmp_42 = None
        tmp_44 = torch.nn.functional.dropout(tmp_43, 0.0, False, False);  tmp_43 = None
        matmul_1 = tmp_44 @ tmp_31;  tmp_44 = tmp_31 = None
        tmp_46 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_47 = tmp_46.reshape(20, 49, 156);  tmp_46 = None
        linear_1 = torch.nn.functional.linear(tmp_47, w_2, w_1);  tmp_47 = w_2 = w_1 = None
        tmp_49 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_50 = tmp_49.reshape(1, 5, 4, 7, 7, 156);  tmp_49 = None
        tmp_51 = tmp_50.permute(0, 1, 3, 2, 4, 5);  tmp_50 = None
        tmp_52 = tmp_51.reshape(1, 35, 28, 156);  tmp_51 = None
        tmp_53 = tmp_52[(slice(None, None, None), slice(1, 33, None), slice(2, 26, None))];  tmp_52 = None
        tmp_54 = tmp_53.reshape(1, 768, 156);  tmp_53 = None
        tmp_55 = tmp_19 + tmp_54;  tmp_19 = tmp_54 = None
        tmp_56 = torch.nn.functional.layer_norm(tmp_55, (156,), w_10, w_9, 1e-06);  w_10 = w_9 = None
        tmp_57 = tmp_56.transpose(1, 2);  tmp_56 = None
        tmp_58 = tmp_57.reshape(1, 156, 32, 24);  tmp_57 = None
        return (tmp_17, tmp_55, tmp_58)
        
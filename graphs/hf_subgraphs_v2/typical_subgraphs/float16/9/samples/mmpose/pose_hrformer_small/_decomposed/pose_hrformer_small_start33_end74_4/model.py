import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1):
        tmp_10 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        tmp_11 = in_0.view(1, 32, -1);  in_0 = None
        tmp_12 = tmp_11.permute(0, 2, 1);  tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (32,), w_7, w_6, 1e-06);  w_7 = w_6 = None
        tmp_14 = tmp_13.view(1, 64, 48, 32);  tmp_13 = None
        tmp_15 = torch.nn.functional.pad(tmp_14, (0, 0, 0, 1, 3, 3), 'constant', None);  tmp_14 = None
        tmp_16 = tmp_15.view(1, 10, 7, 7, 7, 32);  tmp_15 = None
        tmp_17 = tmp_16.permute(0, 1, 3, 2, 4, 5);  tmp_16 = None
        tmp_18 = tmp_17.reshape(-1, 49, 32);  tmp_17 = None
        linear = torch.nn.functional.linear(tmp_18, w_4, w_3);  tmp_18 = w_4 = w_3 = None
        tmp_20 = linear.reshape(70, 49, 3, 1, 32);  linear = None
        tmp_21 = tmp_20.permute(2, 0, 3, 1, 4);  tmp_20 = None
        tmp_22 = tmp_21[0]
        tmp_23 = tmp_21[1]
        tmp_24 = tmp_21[2];  tmp_21 = None
        tmp_25 = tmp_22 * 0.1767766952966369;  tmp_22 = None
        tmp_26 = tmp_23.transpose(-2, -1);  tmp_23 = None
        matmul = tmp_25 @ tmp_26;  tmp_25 = tmp_26 = None
        tmp_28 = w_0.view(-1);  w_0 = None
        tmp_29 = w_5[tmp_28];  w_5 = tmp_28 = None
        tmp_30 = tmp_29.view(49, 49, -1);  tmp_29 = None
        tmp_31 = tmp_30.permute(2, 0, 1);  tmp_30 = None
        tmp_32 = tmp_31.contiguous();  tmp_31 = None
        tmp_33 = tmp_32.unsqueeze(0);  tmp_32 = None
        tmp_34 = matmul + tmp_33;  matmul = tmp_33 = None
        tmp_35 = torch.nn.functional.softmax(tmp_34, -1, _stacklevel = 5);  tmp_34 = None
        tmp_36 = torch.nn.functional.dropout(tmp_35, 0.0, False, False);  tmp_35 = None
        matmul_1 = tmp_36 @ tmp_24;  tmp_36 = tmp_24 = None
        tmp_38 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_39 = tmp_38.reshape(70, 49, 32);  tmp_38 = None
        linear_1 = torch.nn.functional.linear(tmp_39, w_2, w_1);  tmp_39 = w_2 = w_1 = None
        tmp_41 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_42 = tmp_41.reshape(1, 10, 7, 7, 7, 32);  tmp_41 = None
        tmp_43 = tmp_42.permute(0, 1, 3, 2, 4, 5);  tmp_42 = None
        tmp_44 = tmp_43.reshape(1, 70, 49, 32);  tmp_43 = None
        tmp_45 = tmp_44[(slice(None, None, None), slice(3, 67, None), slice(0, 48, None))];  tmp_44 = None
        tmp_46 = tmp_45.reshape(1, 3072, 32);  tmp_45 = None
        tmp_47 = tmp_12 + tmp_46;  tmp_12 = tmp_46 = None
        tmp_48 = torch.nn.functional.layer_norm(tmp_47, (32,), w_9, w_8, 1e-06);  w_9 = w_8 = None
        tmp_49 = tmp_48.transpose(1, 2);  tmp_48 = None
        tmp_50 = tmp_49.reshape(1, 32, 64, 48);  tmp_49 = None
        return (tmp_10, tmp_47, tmp_50)
        
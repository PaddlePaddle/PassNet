import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1, in_2, in_3):
        in_3 += in_0;  in_4 = in_3;  in_3 = in_0 = None
        in_4 += in_2;  tmp_10 = in_4;  in_4 = in_2 = None
        tmp_12 = torch.nn.functional.relu(tmp_10, inplace = True);  tmp_10 = None
        tmp_13 = in_1.view(1, 32, -1);  in_1 = None
        tmp_14 = tmp_13.permute(0, 2, 1);  tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (32,), w_7, w_6, 1e-06);  w_7 = w_6 = None
        tmp_16 = tmp_15.view(1, 64, 48, 32);  tmp_15 = None
        tmp_17 = torch.nn.functional.pad(tmp_16, (0, 0, 0, 1, 3, 3), 'constant', None);  tmp_16 = None
        tmp_18 = tmp_17.view(1, 10, 7, 7, 7, 32);  tmp_17 = None
        tmp_19 = tmp_18.permute(0, 1, 3, 2, 4, 5);  tmp_18 = None
        tmp_20 = tmp_19.reshape(-1, 49, 32);  tmp_19 = None
        linear = torch.nn.functional.linear(tmp_20, w_4, w_3);  tmp_20 = w_4 = w_3 = None
        tmp_22 = linear.reshape(70, 49, 3, 1, 32);  linear = None
        tmp_23 = tmp_22.permute(2, 0, 3, 1, 4);  tmp_22 = None
        tmp_24 = tmp_23[0]
        tmp_25 = tmp_23[1]
        tmp_26 = tmp_23[2];  tmp_23 = None
        tmp_27 = tmp_24 * 0.1767766952966369;  tmp_24 = None
        tmp_28 = tmp_25.transpose(-2, -1);  tmp_25 = None
        matmul = tmp_27 @ tmp_28;  tmp_27 = tmp_28 = None
        tmp_30 = w_0.view(-1);  w_0 = None
        tmp_31 = w_5[tmp_30];  w_5 = tmp_30 = None
        tmp_32 = tmp_31.view(49, 49, -1);  tmp_31 = None
        tmp_33 = tmp_32.permute(2, 0, 1);  tmp_32 = None
        tmp_34 = tmp_33.contiguous();  tmp_33 = None
        tmp_35 = tmp_34.unsqueeze(0);  tmp_34 = None
        tmp_36 = matmul + tmp_35;  matmul = tmp_35 = None
        tmp_37 = torch.nn.functional.softmax(tmp_36, -1, _stacklevel = 5);  tmp_36 = None
        tmp_38 = torch.nn.functional.dropout(tmp_37, 0.0, False, False);  tmp_37 = None
        matmul_1 = tmp_38 @ tmp_26;  tmp_38 = tmp_26 = None
        tmp_40 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_41 = tmp_40.reshape(70, 49, 32);  tmp_40 = None
        linear_1 = torch.nn.functional.linear(tmp_41, w_2, w_1);  tmp_41 = w_2 = w_1 = None
        tmp_43 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_44 = tmp_43.reshape(1, 10, 7, 7, 7, 32);  tmp_43 = None
        tmp_45 = tmp_44.permute(0, 1, 3, 2, 4, 5);  tmp_44 = None
        tmp_46 = tmp_45.reshape(1, 70, 49, 32);  tmp_45 = None
        tmp_47 = tmp_46[(slice(None, None, None), slice(3, 67, None), slice(0, 48, None))];  tmp_46 = None
        tmp_48 = tmp_47.reshape(1, 3072, 32);  tmp_47 = None
        tmp_49 = tmp_14 + tmp_48;  tmp_14 = tmp_48 = None
        tmp_50 = torch.nn.functional.layer_norm(tmp_49, (32,), w_9, w_8, 1e-06);  w_9 = w_8 = None
        tmp_51 = tmp_50.transpose(1, 2);  tmp_50 = None
        tmp_52 = tmp_51.reshape(1, 32, 64, 48);  tmp_51 = None
        return (tmp_12, tmp_49, tmp_52)
        
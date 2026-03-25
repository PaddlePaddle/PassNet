import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, in_0, in_1, in_2, in_3):
        in_3 += in_0;  in_4 = in_3;  in_3 = in_0 = None
        in_4 += in_2;  tmp_11 = in_4;  in_4 = in_2 = None
        tmp_13 = torch.nn.functional.relu(tmp_11, inplace = True);  tmp_11 = None
        tmp_14 = in_1.view(1, 78, -1);  in_1 = None
        tmp_15 = tmp_14.permute(0, 2, 1);  tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (78,), w_8, w_7, 1e-06);  w_8 = w_7 = None
        tmp_17 = tmp_16.view(1, 64, 48, 78);  tmp_16 = None
        tmp_18 = torch.nn.functional.pad(tmp_17, (0, 0, 0, 1, 3, 3), 'constant', None);  tmp_17 = None
        tmp_19 = tmp_18.view(1, 10, 7, 7, 7, 78);  tmp_18 = None
        tmp_20 = tmp_19.permute(0, 1, 3, 2, 4, 5);  tmp_19 = None
        tmp_21 = tmp_20.reshape(-1, 49, 78);  tmp_20 = None
        linear = torch.nn.functional.linear(tmp_21, w_4, w_3);  tmp_21 = w_4 = w_3 = None
        tmp_23 = linear.reshape(70, 49, 3, 2, 39);  linear = None
        tmp_24 = tmp_23.permute(2, 0, 3, 1, 4);  tmp_23 = None
        tmp_25 = tmp_24[0]
        tmp_26 = tmp_24[1]
        tmp_27 = tmp_24[2];  tmp_24 = None
        item = w_6.item();  w_6 = None
        tmp_29 = tmp_25 * item;  tmp_25 = item = None
        tmp_30 = tmp_26.transpose(-2, -1);  tmp_26 = None
        matmul = tmp_29 @ tmp_30;  tmp_29 = tmp_30 = None
        tmp_32 = w_0.view(-1);  w_0 = None
        tmp_33 = w_5[tmp_32];  w_5 = tmp_32 = None
        tmp_34 = tmp_33.view(49, 49, -1);  tmp_33 = None
        tmp_35 = tmp_34.permute(2, 0, 1);  tmp_34 = None
        tmp_36 = tmp_35.contiguous();  tmp_35 = None
        tmp_37 = tmp_36.unsqueeze(0);  tmp_36 = None
        tmp_38 = matmul + tmp_37;  matmul = tmp_37 = None
        tmp_39 = torch.nn.functional.softmax(tmp_38, -1, _stacklevel = 5);  tmp_38 = None
        tmp_40 = torch.nn.functional.dropout(tmp_39, 0.0, False, False);  tmp_39 = None
        matmul_1 = tmp_40 @ tmp_27;  tmp_40 = tmp_27 = None
        tmp_42 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_43 = tmp_42.reshape(70, 49, 78);  tmp_42 = None
        linear_1 = torch.nn.functional.linear(tmp_43, w_2, w_1);  tmp_43 = w_2 = w_1 = None
        tmp_45 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_46 = tmp_45.reshape(1, 10, 7, 7, 7, 78);  tmp_45 = None
        tmp_47 = tmp_46.permute(0, 1, 3, 2, 4, 5);  tmp_46 = None
        tmp_48 = tmp_47.reshape(1, 70, 49, 78);  tmp_47 = None
        tmp_49 = tmp_48[(slice(None, None, None), slice(3, 67, None), slice(0, 48, None))];  tmp_48 = None
        tmp_50 = tmp_49.reshape(1, 3072, 78);  tmp_49 = None
        tmp_51 = tmp_15 + tmp_50;  tmp_15 = tmp_50 = None
        tmp_52 = torch.nn.functional.layer_norm(tmp_51, (78,), w_10, w_9, 1e-06);  w_10 = w_9 = None
        tmp_53 = tmp_52.transpose(1, 2);  tmp_52 = None
        tmp_54 = tmp_53.reshape(1, 78, 64, 48);  tmp_53 = None
        return (tmp_13, tmp_51, tmp_54)
        
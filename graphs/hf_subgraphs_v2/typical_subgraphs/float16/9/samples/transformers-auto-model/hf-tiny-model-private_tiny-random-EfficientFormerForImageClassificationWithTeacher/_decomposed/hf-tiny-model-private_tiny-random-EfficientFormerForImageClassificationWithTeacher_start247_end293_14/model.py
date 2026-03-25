import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, in_0, in_1, in_2):
        tmp_21 = torch.nn.functional.dropout(in_0, 0.1, False, False);  in_0 = None
        tmp_22 = in_2 * tmp_21;  in_2 = tmp_21 = None
        tmp_23 = in_1 + tmp_22;  in_1 = tmp_22 = None
        tmp_24 = tmp_23.flatten(2);  tmp_23 = None
        tmp_25 = tmp_24.transpose(1, 2);  tmp_24 = None
        tmp_26 = torch.nn.functional.layer_norm(tmp_25, (448,), w_5, w_4, 1e-12);  w_5 = w_4 = None
        linear = torch.nn.functional.linear(tmp_26, w_16, w_15);  tmp_26 = w_16 = w_15 = None
        tmp_28 = linear.reshape(1, 49, 8, -1);  linear = None
        split = tmp_28.split([32, 32, 128], dim = 3);  tmp_28 = None
        tmp_30 = split[0]
        tmp_31 = split[1]
        tmp_32 = split[2];  split = None
        tmp_33 = tmp_30.permute(0, 2, 1, 3);  tmp_30 = None
        tmp_34 = tmp_31.permute(0, 2, 1, 3);  tmp_31 = None
        tmp_35 = tmp_32.permute(0, 2, 1, 3);  tmp_32 = None
        tmp_36 = w_12.to(device(type='cuda', index=0));  w_12 = None
        tmp_37 = tmp_34.transpose(-2, -1);  tmp_34 = None
        matmul = torch.matmul(tmp_33, tmp_37);  tmp_33 = tmp_37 = None
        tmp_39 = matmul * 0.1767766952966369;  matmul = None
        tmp_40 = tmp_39 + tmp_36;  tmp_39 = None
        tmp_41 = tmp_40.softmax(dim = -1);  tmp_40 = None
        matmul_1 = torch.matmul(tmp_41, tmp_35);  tmp_41 = tmp_35 = None
        tmp_43 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_44 = tmp_43.reshape(1, 49, 1024);  tmp_43 = None
        linear_1 = torch.nn.functional.linear(tmp_44, w_14, w_13);  tmp_44 = w_14 = w_13 = None
        tmp_46 = w_17.unsqueeze(0);  w_17 = None
        tmp_47 = tmp_46.unsqueeze(0);  tmp_46 = None
        tmp_48 = tmp_47 * linear_1;  tmp_47 = linear_1 = None
        tmp_49 = tmp_25 + tmp_48;  tmp_25 = tmp_48 = None
        tmp_50 = w_18.unsqueeze(0);  w_18 = None
        tmp_51 = tmp_50.unsqueeze(0);  tmp_50 = None
        tmp_52 = torch.nn.functional.layer_norm(tmp_49, (448,), w_7, w_6, 1e-12);  w_7 = w_6 = None
        linear_2 = torch.nn.functional.linear(tmp_52, w_9, w_8);  tmp_52 = w_9 = w_8 = None
        tmp_54 = torch.nn.functional.gelu(linear_2);  linear_2 = None
        tmp_55 = torch.nn.functional.dropout(tmp_54, 0.1, False, False);  tmp_54 = None
        linear_3 = torch.nn.functional.linear(tmp_55, w_11, w_10);  tmp_55 = w_11 = w_10 = None
        tmp_57 = torch.nn.functional.dropout(linear_3, 0.1, False, False);  linear_3 = None
        tmp_58 = tmp_51 * tmp_57;  tmp_51 = tmp_57 = None
        tmp_59 = tmp_49 + tmp_58;  tmp_49 = tmp_58 = None
        tmp_60 = torch.nn.functional.layer_norm(tmp_59, (448,), w_20, w_19, 1e-12);  tmp_59 = w_20 = w_19 = None
        tmp_61 = tmp_60.mean(-2)
        linear_4 = torch.nn.functional.linear(tmp_61, w_1, w_0);  tmp_61 = w_1 = w_0 = None
        tmp_63 = tmp_60.mean(-2);  tmp_60 = None
        linear_5 = torch.nn.functional.linear(tmp_63, w_3, w_2);  tmp_63 = w_3 = w_2 = None
        tmp_65 = linear_4 + linear_5
        tmp_66 = tmp_65 / 2;  tmp_65 = None
        return (tmp_36, linear_4, linear_5, tmp_66)
        
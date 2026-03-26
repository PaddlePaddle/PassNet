import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23):
        tmp_21 = torch.nn.functional.dropout(in_21, 0.1, False, False);  in_21 = None
        tmp_22 = in_23 * tmp_21;  in_23 = tmp_21 = None
        tmp_23 = in_22 + tmp_22;  in_22 = tmp_22 = None
        tmp_24 = tmp_23.flatten(2);  tmp_23 = None
        tmp_25 = tmp_24.transpose(1, 2);  tmp_24 = None
        tmp_26 = torch.nn.functional.layer_norm(tmp_25, (448,), in_5, in_4, 1e-12);  in_5 = in_4 = None
        linear = torch.nn.functional.linear(tmp_26, in_16, in_15);  tmp_26 = in_16 = in_15 = None
        tmp_28 = linear.reshape(8, 49, 8, -1);  linear = None
        split = tmp_28.split([32, 32, 128], dim = 3);  tmp_28 = None
        tmp_30 = split[0]
        tmp_31 = split[1]
        tmp_32 = split[2];  split = None
        tmp_33 = tmp_30.permute(0, 2, 1, 3);  tmp_30 = None
        tmp_34 = tmp_31.permute(0, 2, 1, 3);  tmp_31 = None
        tmp_35 = tmp_32.permute(0, 2, 1, 3);  tmp_32 = None
        tmp_36 = in_12.to(device(type='cuda', index=0));  in_12 = None
        tmp_37 = tmp_34.transpose(-2, -1);  tmp_34 = None
        matmul = torch.matmul(tmp_33, tmp_37);  tmp_33 = tmp_37 = None
        tmp_39 = matmul * 0.1767766952966369;  matmul = None
        tmp_40 = tmp_39 + tmp_36;  tmp_39 = None
        tmp_41 = tmp_40.softmax(dim = -1);  tmp_40 = None
        matmul_1 = torch.matmul(tmp_41, tmp_35);  tmp_41 = tmp_35 = None
        tmp_43 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_44 = tmp_43.reshape(8, 49, 1024);  tmp_43 = None
        linear_1 = torch.nn.functional.linear(tmp_44, in_14, in_13);  tmp_44 = in_14 = in_13 = None
        tmp_46 = in_17.unsqueeze(0);  in_17 = None
        tmp_47 = tmp_46.unsqueeze(0);  tmp_46 = None
        tmp_48 = tmp_47 * linear_1;  tmp_47 = linear_1 = None
        tmp_49 = tmp_25 + tmp_48;  tmp_25 = tmp_48 = None
        tmp_50 = in_18.unsqueeze(0);  in_18 = None
        tmp_51 = tmp_50.unsqueeze(0);  tmp_50 = None
        tmp_52 = torch.nn.functional.layer_norm(tmp_49, (448,), in_7, in_6, 1e-12);  in_7 = in_6 = None
        linear_2 = torch.nn.functional.linear(tmp_52, in_9, in_8);  tmp_52 = in_9 = in_8 = None
        tmp_54 = torch.nn.functional.gelu(linear_2);  linear_2 = None
        tmp_55 = torch.nn.functional.dropout(tmp_54, 0.1, False, False);  tmp_54 = None
        linear_3 = torch.nn.functional.linear(tmp_55, in_11, in_10);  tmp_55 = in_11 = in_10 = None
        tmp_57 = torch.nn.functional.dropout(linear_3, 0.1, False, False);  linear_3 = None
        tmp_58 = tmp_51 * tmp_57;  tmp_51 = tmp_57 = None
        tmp_59 = tmp_49 + tmp_58;  tmp_49 = tmp_58 = None
        tmp_60 = torch.nn.functional.layer_norm(tmp_59, (448,), in_20, in_19, 1e-12);  tmp_59 = in_20 = in_19 = None
        tmp_61 = tmp_60.mean(-2)
        linear_4 = torch.nn.functional.linear(tmp_61, in_1, in_0);  tmp_61 = in_1 = in_0 = None
        tmp_63 = tmp_60.mean(-2);  tmp_60 = None
        linear_5 = torch.nn.functional.linear(tmp_63, in_3, in_2);  tmp_63 = in_3 = in_2 = None
        tmp_65 = linear_4 + linear_5
        tmp_66 = tmp_65 / 2;  tmp_65 = None
        return (tmp_36, linear_4, linear_5, tmp_66)
        
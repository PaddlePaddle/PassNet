import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19):
        tmp_17 = torch.nn.functional.dropout(in_17, 0.1, False, False);  in_17 = None
        tmp_18 = in_19 * tmp_17;  in_19 = tmp_17 = None
        tmp_19 = in_18 + tmp_18;  in_18 = tmp_18 = None
        tmp_20 = tmp_19.flatten(2);  tmp_19 = None
        tmp_21 = tmp_20.transpose(1, 2);  tmp_20 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (448,), in_1, in_0, 1e-12);  in_1 = in_0 = None
        linear = torch.nn.functional.linear(tmp_22, in_12, in_11);  tmp_22 = in_12 = in_11 = None
        tmp_24 = linear.reshape(16, 49, 8, -1);  linear = None
        split = tmp_24.split([32, 32, 128], dim = 3);  tmp_24 = None
        tmp_26 = split[0]
        tmp_27 = split[1]
        tmp_28 = split[2];  split = None
        tmp_29 = tmp_26.permute(0, 2, 1, 3);  tmp_26 = None
        tmp_30 = tmp_27.permute(0, 2, 1, 3);  tmp_27 = None
        tmp_31 = tmp_28.permute(0, 2, 1, 3);  tmp_28 = None
        tmp_32 = in_8.to(device(type='cuda', index=0));  in_8 = None
        tmp_33 = tmp_30.transpose(-2, -1);  tmp_30 = None
        matmul = torch.matmul(tmp_29, tmp_33);  tmp_29 = tmp_33 = None
        tmp_35 = matmul * 0.1767766952966369;  matmul = None
        tmp_36 = tmp_35 + tmp_32;  tmp_35 = None
        tmp_37 = tmp_36.softmax(dim = -1);  tmp_36 = None
        matmul_1 = torch.matmul(tmp_37, tmp_31);  tmp_37 = tmp_31 = None
        tmp_39 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_40 = tmp_39.reshape(16, 49, 1024);  tmp_39 = None
        linear_1 = torch.nn.functional.linear(tmp_40, in_10, in_9);  tmp_40 = in_10 = in_9 = None
        tmp_42 = in_13.unsqueeze(0);  in_13 = None
        tmp_43 = tmp_42.unsqueeze(0);  tmp_42 = None
        tmp_44 = tmp_43 * linear_1;  tmp_43 = linear_1 = None
        tmp_45 = tmp_21 + tmp_44;  tmp_21 = tmp_44 = None
        tmp_46 = in_14.unsqueeze(0);  in_14 = None
        tmp_47 = tmp_46.unsqueeze(0);  tmp_46 = None
        tmp_48 = torch.nn.functional.layer_norm(tmp_45, (448,), in_3, in_2, 1e-12);  in_3 = in_2 = None
        linear_2 = torch.nn.functional.linear(tmp_48, in_5, in_4);  tmp_48 = in_5 = in_4 = None
        tmp_50 = torch.nn.functional.gelu(linear_2);  linear_2 = None
        tmp_51 = torch.nn.functional.dropout(tmp_50, 0.1, False, False);  tmp_50 = None
        linear_3 = torch.nn.functional.linear(tmp_51, in_7, in_6);  tmp_51 = in_7 = in_6 = None
        tmp_53 = torch.nn.functional.dropout(linear_3, 0.1, False, False);  linear_3 = None
        tmp_54 = tmp_47 * tmp_53;  tmp_47 = tmp_53 = None
        tmp_55 = tmp_45 + tmp_54;  tmp_45 = tmp_54 = None
        tmp_56 = torch.nn.functional.layer_norm(tmp_55, (448,), in_16, in_15, 1e-12);  tmp_55 = in_16 = in_15 = None
        return (tmp_32, tmp_56)
        
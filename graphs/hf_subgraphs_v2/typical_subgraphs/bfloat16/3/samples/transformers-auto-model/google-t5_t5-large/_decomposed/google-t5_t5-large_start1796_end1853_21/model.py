import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14):
        tmp_11 = in_12.view(1, -1, 1024);  in_12 = None
        linear = torch.nn.functional.linear(tmp_11, in_1, None);  tmp_11 = in_1 = None
        tmp_13 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_14 = in_13 + tmp_13;  in_13 = tmp_13 = None
        tmp_15 = in_11[-1];  in_11 = None
        tmp_16 = tmp_15 + 1;  tmp_15 = tmp_16 = None
        tmp_17 = tmp_14.to(torch.float32)
        tmp_18 = tmp_17.pow(2);  tmp_17 = None
        tmp_19 = tmp_18.mean(-1, keepdim = True);  tmp_18 = None
        tmp_20 = tmp_19 + 1e-06;  tmp_19 = None
        tmp_21 = torch.rsqrt(tmp_20);  tmp_20 = None
        tmp_22 = tmp_14 * tmp_21;  tmp_21 = None
        tmp_23 = in_6 * tmp_22;  in_6 = tmp_22 = None
        to_1 = tmp_23.to(torch.bfloat16);  tmp_23 = None
        linear_1 = torch.nn.functional.linear(to_1, in_4, None);  to_1 = in_4 = None
        tmp_25 = linear_1.view(1, -1, 16, 64);  linear_1 = None
        tmp_26 = tmp_25.transpose(1, 2);  tmp_25 = None
        linear_2 = torch.nn.functional.linear(in_0, in_2, None);  in_2 = None
        linear_3 = torch.nn.functional.linear(in_0, in_5, None);  in_0 = in_5 = None
        tmp_29 = linear_2.view(1, -1, 16, 64);  linear_2 = None
        tmp_30 = tmp_29.transpose(1, 2);  tmp_29 = None
        tmp_31 = linear_3.view(1, -1, 16, 64);  linear_3 = None
        tmp_32 = tmp_31.transpose(1, 2);  tmp_31 = None
        tmp_33 = tmp_30.transpose(3, 2)
        matmul = torch.matmul(tmp_26, tmp_33);  tmp_26 = tmp_33 = None
        matmul += in_14;  tmp_35 = matmul;  matmul = in_14 = None
        tmp_36 = tmp_35.float()
        tmp_37 = torch.nn.functional.softmax(tmp_36, dim = -1);  tmp_36 = None
        tmp_38 = tmp_37.type_as(tmp_35);  tmp_37 = tmp_35 = None
        tmp_39 = torch.nn.functional.dropout(tmp_38, p = 0.1, training = False);  tmp_38 = None
        matmul_1 = torch.matmul(tmp_39, tmp_32);  tmp_39 = None
        tmp_41 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_42 = tmp_41.contiguous();  tmp_41 = None
        tmp_43 = tmp_42.view(1, -1, 1024);  tmp_42 = None
        linear_4 = torch.nn.functional.linear(tmp_43, in_3, None);  tmp_43 = in_3 = None
        tmp_45 = torch.nn.functional.dropout(linear_4, 0.1, False, False);  linear_4 = None
        tmp_46 = tmp_14 + tmp_45;  tmp_14 = tmp_45 = None
        tmp_47 = tmp_46.to(torch.float32)
        tmp_48 = tmp_47.pow(2);  tmp_47 = None
        tmp_49 = tmp_48.mean(-1, keepdim = True);  tmp_48 = None
        tmp_50 = tmp_49 + 1e-06;  tmp_49 = None
        tmp_51 = torch.rsqrt(tmp_50);  tmp_50 = None
        tmp_52 = tmp_46 * tmp_51;  tmp_51 = None
        tmp_53 = in_9 * tmp_52;  in_9 = tmp_52 = None
        to_7 = tmp_53.to(torch.bfloat16);  tmp_53 = None
        linear_5 = torch.nn.functional.linear(to_7, in_7, None);  to_7 = in_7 = None
        tmp_55 = torch.nn.functional.relu(linear_5, inplace = False);  linear_5 = None
        tmp_56 = torch.nn.functional.dropout(tmp_55, 0.1, False, False);  tmp_55 = None
        linear_6 = torch.nn.functional.linear(tmp_56, in_8, None);  tmp_56 = in_8 = None
        tmp_58 = torch.nn.functional.dropout(linear_6, 0.1, False, False);  linear_6 = None
        tmp_59 = tmp_46 + tmp_58;  tmp_46 = tmp_58 = None
        tmp_60 = tmp_59.to(torch.float32)
        tmp_61 = tmp_60.pow(2);  tmp_60 = None
        tmp_62 = tmp_61.mean(-1, keepdim = True);  tmp_61 = None
        tmp_63 = tmp_62 + 1e-06;  tmp_62 = None
        tmp_64 = torch.rsqrt(tmp_63);  tmp_63 = None
        tmp_65 = tmp_59 * tmp_64;  tmp_59 = tmp_64 = None
        tmp_66 = in_10 * tmp_65;  in_10 = tmp_65 = None
        tmp_67 = torch.nn.functional.dropout(tmp_66, 0.1, False, False);  tmp_66 = None
        return (tmp_30, tmp_32, tmp_67)
        
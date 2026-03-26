import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1, in_2, in_3, in_4):
        tmp_10 = in_0.view(1, -1, 512);  in_0 = None
        linear = torch.nn.functional.linear(tmp_10, w_0, None);  tmp_10 = w_0 = None
        tmp_12 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_13 = in_3 + tmp_12;  in_3 = tmp_12 = None
        tmp_14 = in_1[-1];  in_1 = None
        tmp_15 = tmp_14 + 1;  tmp_14 = tmp_15 = None
        tmp_16 = tmp_13.to(torch.float32)
        tmp_17 = tmp_16.pow(2);  tmp_16 = None
        tmp_18 = tmp_17.mean(-1, keepdim = True);  tmp_17 = None
        tmp_19 = tmp_18 + 1e-06;  tmp_18 = None
        tmp_20 = torch.rsqrt(tmp_19);  tmp_19 = None
        tmp_21 = tmp_13 * tmp_20;  tmp_20 = None
        tmp_22 = w_5 * tmp_21;  w_5 = tmp_21 = None
        to_1 = tmp_22.to(torch.float16);  tmp_22 = None
        linear_1 = torch.nn.functional.linear(to_1, w_3, None);  to_1 = w_3 = None
        tmp_24 = linear_1.view(1, -1, 8, 64);  linear_1 = None
        tmp_25 = tmp_24.transpose(1, 2);  tmp_24 = None
        linear_2 = torch.nn.functional.linear(in_2, w_1, None);  w_1 = None
        linear_3 = torch.nn.functional.linear(in_2, w_4, None);  in_2 = w_4 = None
        tmp_28 = linear_2.view(1, -1, 8, 64);  linear_2 = None
        tmp_29 = tmp_28.transpose(1, 2);  tmp_28 = None
        tmp_30 = linear_3.view(1, -1, 8, 64);  linear_3 = None
        tmp_31 = tmp_30.transpose(1, 2);  tmp_30 = None
        tmp_32 = tmp_29.transpose(3, 2);  tmp_29 = None
        matmul = torch.matmul(tmp_25, tmp_32);  tmp_25 = tmp_32 = None
        matmul += in_4;  tmp_34 = matmul;  matmul = in_4 = None
        tmp_35 = tmp_34.float()
        tmp_36 = torch.nn.functional.softmax(tmp_35, dim = -1);  tmp_35 = None
        tmp_37 = tmp_36.type_as(tmp_34);  tmp_36 = tmp_34 = None
        tmp_38 = torch.nn.functional.dropout(tmp_37, p = 0.1, training = False);  tmp_37 = None
        matmul_1 = torch.matmul(tmp_38, tmp_31);  tmp_38 = tmp_31 = None
        tmp_40 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_41 = tmp_40.contiguous();  tmp_40 = None
        tmp_42 = tmp_41.view(1, -1, 512);  tmp_41 = None
        linear_4 = torch.nn.functional.linear(tmp_42, w_2, None);  tmp_42 = w_2 = None
        tmp_44 = torch.nn.functional.dropout(linear_4, 0.1, False, False);  linear_4 = None
        tmp_45 = tmp_13 + tmp_44;  tmp_13 = tmp_44 = None
        tmp_46 = tmp_45.to(torch.float32)
        tmp_47 = tmp_46.pow(2);  tmp_46 = None
        tmp_48 = tmp_47.mean(-1, keepdim = True);  tmp_47 = None
        tmp_49 = tmp_48 + 1e-06;  tmp_48 = None
        tmp_50 = torch.rsqrt(tmp_49);  tmp_49 = None
        tmp_51 = tmp_45 * tmp_50;  tmp_50 = None
        tmp_52 = w_8 * tmp_51;  w_8 = tmp_51 = None
        to_7 = tmp_52.to(torch.float16);  tmp_52 = None
        linear_5 = torch.nn.functional.linear(to_7, w_6, None);  to_7 = w_6 = None
        tmp_54 = torch.nn.functional.relu(linear_5, inplace = False);  linear_5 = None
        tmp_55 = torch.nn.functional.dropout(tmp_54, 0.1, False, False);  tmp_54 = None
        linear_6 = torch.nn.functional.linear(tmp_55, w_7, None);  tmp_55 = w_7 = None
        tmp_57 = torch.nn.functional.dropout(linear_6, 0.1, False, False);  linear_6 = None
        tmp_58 = tmp_45 + tmp_57;  tmp_45 = tmp_57 = None
        tmp_59 = tmp_58.to(torch.float32)
        tmp_60 = tmp_59.pow(2);  tmp_59 = None
        tmp_61 = tmp_60.mean(-1, keepdim = True);  tmp_60 = None
        tmp_62 = tmp_61 + 1e-06;  tmp_61 = None
        tmp_63 = torch.rsqrt(tmp_62);  tmp_62 = None
        tmp_64 = tmp_58 * tmp_63;  tmp_58 = tmp_63 = None
        tmp_65 = w_9 * tmp_64;  w_9 = tmp_64 = None
        tmp_66 = torch.nn.functional.dropout(tmp_65, 0.1, False, False);  tmp_65 = None
        return (tmp_66,)
        
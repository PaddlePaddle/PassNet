import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, w_34, w_35, w_36, w_37, w_38, w_39, w_40, w_41, in_0, in_1, in_2, in_3, in_4):
        linear = torch.nn.functional.linear(in_0, w_5, w_4);  w_5 = w_4 = None
        tmp_43 = linear.view(1, -1, 4, 32);  linear = None
        tmp_44 = tmp_43.transpose(1, 2);  tmp_43 = None
        tmp_45 = in_2.transpose(-1, -2);  in_2 = None
        matmul = torch.matmul(in_4, tmp_45);  in_4 = tmp_45 = None
        tmp_47 = matmul / 5.656854249492381;  matmul = None
        tmp_48 = tmp_47 + in_1;  tmp_47 = in_1 = None
        tmp_49 = torch.nn.functional.softmax(tmp_48, dim = -1);  tmp_48 = None
        tmp_50 = torch.nn.functional.dropout(tmp_49, 0.1, False, False);  tmp_49 = None
        matmul_1 = torch.matmul(tmp_50, tmp_44);  tmp_50 = tmp_44 = None
        tmp_52 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_53 = tmp_52.contiguous();  tmp_52 = None
        tmp_54 = tmp_53.view((1, 11, 128));  tmp_53 = None
        linear_1 = torch.nn.functional.linear(tmp_54, w_3, w_2);  tmp_54 = w_3 = w_2 = None
        tmp_56 = linear_1 + in_3;  linear_1 = in_3 = None
        tmp_57 = tmp_56 * w_1;  tmp_56 = w_1 = None
        tmp_58 = tmp_57 + w_0;  tmp_57 = w_0 = None
        linear_2 = torch.nn.functional.linear(tmp_58, w_7, w_6);  w_7 = w_6 = None
        tmp_60 = torch.nn.functional.relu(linear_2, inplace = False);  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_60, w_11, w_10);  tmp_60 = w_11 = w_10 = None
        tmp_62 = linear_3 + tmp_58;  linear_3 = tmp_58 = None
        tmp_63 = tmp_62 * w_9;  tmp_62 = w_9 = None
        tmp_64 = tmp_63 + w_8;  tmp_63 = w_8 = None
        linear_4 = torch.nn.functional.linear(tmp_64, w_13, w_12);  w_13 = w_12 = None
        tmp_66 = torch.nn.functional.relu(linear_4, inplace = False);  linear_4 = None
        linear_5 = torch.nn.functional.linear(tmp_66, w_17, w_16);  tmp_66 = w_17 = w_16 = None
        tmp_68 = linear_5 + tmp_64;  linear_5 = tmp_64 = None
        tmp_69 = tmp_68 * w_15;  tmp_68 = w_15 = None
        tmp_70 = tmp_69 + w_14;  tmp_69 = w_14 = None
        linear_6 = torch.nn.functional.linear(tmp_70, w_19, w_18);  w_19 = w_18 = None
        tmp_72 = torch.nn.functional.relu(linear_6, inplace = False);  linear_6 = None
        linear_7 = torch.nn.functional.linear(tmp_72, w_23, w_22);  tmp_72 = w_23 = w_22 = None
        tmp_74 = linear_7 + tmp_70;  linear_7 = tmp_70 = None
        tmp_75 = tmp_74 * w_21;  tmp_74 = w_21 = None
        tmp_76 = tmp_75 + w_20;  tmp_75 = w_20 = None
        linear_8 = torch.nn.functional.linear(tmp_76, w_25, w_24);  w_25 = w_24 = None
        tmp_78 = torch.nn.functional.relu(linear_8, inplace = False);  linear_8 = None
        linear_9 = torch.nn.functional.linear(tmp_78, w_33, w_32);  tmp_78 = w_33 = w_32 = None
        tmp_80 = linear_9 + tmp_76;  linear_9 = tmp_76 = None
        tmp_81 = tmp_80 * w_27;  tmp_80 = w_27 = None
        tmp_82 = tmp_81 + w_26;  tmp_81 = w_26 = None
        linear_10 = torch.nn.functional.linear(tmp_82, w_31, w_30);  tmp_82 = w_31 = w_30 = None
        tmp_84 = torch.nn.functional.dropout(linear_10, 0.0, False, False);  linear_10 = None
        tmp_85 = tmp_84 + in_0;  tmp_84 = in_0 = None
        tmp_86 = tmp_85 * w_29;  tmp_85 = w_29 = None
        tmp_87 = tmp_86 + w_28;  tmp_86 = w_28 = None
        tmp_88 = torch.tensor(1000);  tmp_88 = None
        linear_11 = torch.nn.functional.linear(tmp_87, w_41, w_40);  w_41 = w_40 = None
        tmp_90 = linear_11 * w_39;  linear_11 = w_39 = None
        tmp_91 = tmp_90 + w_38;  tmp_90 = w_38 = None
        linear_12 = torch.nn.functional.linear(tmp_87, w_37, w_36);  w_37 = w_36 = None
        tmp_93 = linear_12 * w_35;  linear_12 = w_35 = None
        tmp_94 = tmp_93 + w_34;  tmp_93 = w_34 = None
        return (tmp_91, tmp_94, tmp_87)
        
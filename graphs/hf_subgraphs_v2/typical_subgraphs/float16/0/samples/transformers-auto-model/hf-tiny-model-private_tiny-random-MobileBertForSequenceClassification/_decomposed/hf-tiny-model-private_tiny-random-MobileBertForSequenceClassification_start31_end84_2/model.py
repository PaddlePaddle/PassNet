import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46):
        linear = torch.nn.functional.linear(in_42, in_5, in_4);  in_5 = in_4 = None
        tmp_43 = linear.view(1, -1, 4, 32);  linear = None
        tmp_44 = tmp_43.transpose(1, 2);  tmp_43 = None
        tmp_45 = in_44.transpose(-1, -2);  in_44 = None
        matmul = torch.matmul(in_46, tmp_45);  in_46 = tmp_45 = None
        tmp_47 = matmul / 5.656854249492381;  matmul = None
        tmp_48 = tmp_47 + in_43;  tmp_47 = in_43 = None
        tmp_49 = torch.nn.functional.softmax(tmp_48, dim = -1);  tmp_48 = None
        tmp_50 = torch.nn.functional.dropout(tmp_49, 0.1, False, False);  tmp_49 = None
        matmul_1 = torch.matmul(tmp_50, tmp_44);  tmp_50 = tmp_44 = None
        tmp_52 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_53 = tmp_52.contiguous();  tmp_52 = None
        tmp_54 = tmp_53.view((1, 64, 128));  tmp_53 = None
        linear_1 = torch.nn.functional.linear(tmp_54, in_3, in_2);  tmp_54 = in_3 = in_2 = None
        tmp_56 = linear_1 + in_45;  linear_1 = in_45 = None
        tmp_57 = tmp_56 * in_1;  tmp_56 = in_1 = None
        tmp_58 = tmp_57 + in_0;  tmp_57 = in_0 = None
        linear_2 = torch.nn.functional.linear(tmp_58, in_7, in_6);  in_7 = in_6 = None
        tmp_60 = torch.nn.functional.gelu(linear_2);  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_60, in_11, in_10);  tmp_60 = in_11 = in_10 = None
        tmp_62 = linear_3 + tmp_58;  linear_3 = tmp_58 = None
        tmp_63 = tmp_62 * in_9;  tmp_62 = in_9 = None
        tmp_64 = tmp_63 + in_8;  tmp_63 = in_8 = None
        linear_4 = torch.nn.functional.linear(tmp_64, in_13, in_12);  in_13 = in_12 = None
        tmp_66 = torch.nn.functional.gelu(linear_4);  linear_4 = None
        linear_5 = torch.nn.functional.linear(tmp_66, in_17, in_16);  tmp_66 = in_17 = in_16 = None
        tmp_68 = linear_5 + tmp_64;  linear_5 = tmp_64 = None
        tmp_69 = tmp_68 * in_15;  tmp_68 = in_15 = None
        tmp_70 = tmp_69 + in_14;  tmp_69 = in_14 = None
        linear_6 = torch.nn.functional.linear(tmp_70, in_19, in_18);  in_19 = in_18 = None
        tmp_72 = torch.nn.functional.gelu(linear_6);  linear_6 = None
        linear_7 = torch.nn.functional.linear(tmp_72, in_23, in_22);  tmp_72 = in_23 = in_22 = None
        tmp_74 = linear_7 + tmp_70;  linear_7 = tmp_70 = None
        tmp_75 = tmp_74 * in_21;  tmp_74 = in_21 = None
        tmp_76 = tmp_75 + in_20;  tmp_75 = in_20 = None
        linear_8 = torch.nn.functional.linear(tmp_76, in_25, in_24);  in_25 = in_24 = None
        tmp_78 = torch.nn.functional.gelu(linear_8);  linear_8 = None
        linear_9 = torch.nn.functional.linear(tmp_78, in_33, in_32);  tmp_78 = in_33 = in_32 = None
        tmp_80 = linear_9 + tmp_76;  linear_9 = tmp_76 = None
        tmp_81 = tmp_80 * in_27;  tmp_80 = in_27 = None
        tmp_82 = tmp_81 + in_26;  tmp_81 = in_26 = None
        linear_10 = torch.nn.functional.linear(tmp_82, in_31, in_30);  tmp_82 = in_31 = in_30 = None
        tmp_84 = torch.nn.functional.dropout(linear_10, 0.1, False, False);  linear_10 = None
        tmp_85 = tmp_84 + in_42;  tmp_84 = in_42 = None
        tmp_86 = tmp_85 * in_29;  tmp_85 = in_29 = None
        tmp_87 = tmp_86 + in_28;  tmp_86 = in_28 = None
        tmp_88 = torch.tensor(1000);  tmp_88 = None
        linear_11 = torch.nn.functional.linear(tmp_87, in_41, in_40);  in_41 = in_40 = None
        tmp_90 = linear_11 * in_39;  linear_11 = in_39 = None
        tmp_91 = tmp_90 + in_38;  tmp_90 = in_38 = None
        linear_12 = torch.nn.functional.linear(tmp_87, in_37, in_36);  in_37 = in_36 = None
        tmp_93 = linear_12 * in_35;  linear_12 = in_35 = None
        tmp_94 = tmp_93 + in_34;  tmp_93 = in_34 = None
        return (tmp_91, tmp_94, tmp_87)
        
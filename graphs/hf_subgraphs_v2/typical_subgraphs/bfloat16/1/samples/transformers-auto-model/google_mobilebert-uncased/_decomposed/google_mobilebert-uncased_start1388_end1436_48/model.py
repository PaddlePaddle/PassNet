import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38):
        linear = torch.nn.functional.linear(in_37, in_5, in_4);  in_5 = in_4 = None
        tmp_35 = linear.view(1, -1, 4, 32);  linear = None
        tmp_36 = tmp_35.transpose(1, 2);  tmp_35 = None
        tmp_37 = in_35.transpose(-1, -2);  in_35 = None
        matmul = torch.matmul(in_38, tmp_37);  in_38 = tmp_37 = None
        tmp_39 = matmul / 5.656854249492381;  matmul = None
        tmp_40 = tmp_39 + in_34;  tmp_39 = in_34 = None
        tmp_41 = torch.nn.functional.softmax(tmp_40, dim = -1);  tmp_40 = None
        tmp_42 = torch.nn.functional.dropout(tmp_41, 0.1, False, False);  tmp_41 = None
        matmul_1 = torch.matmul(tmp_42, tmp_36);  tmp_42 = tmp_36 = None
        tmp_44 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_45 = tmp_44.contiguous();  tmp_44 = None
        tmp_46 = tmp_45.view((1, 512, 128));  tmp_45 = None
        linear_1 = torch.nn.functional.linear(tmp_46, in_3, in_2);  tmp_46 = in_3 = in_2 = None
        tmp_48 = linear_1 + in_36;  linear_1 = in_36 = None
        tmp_49 = tmp_48 * in_1;  tmp_48 = in_1 = None
        tmp_50 = tmp_49 + in_0;  tmp_49 = in_0 = None
        linear_2 = torch.nn.functional.linear(tmp_50, in_7, in_6);  in_7 = in_6 = None
        tmp_52 = torch.nn.functional.relu(linear_2, inplace = False);  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_52, in_11, in_10);  tmp_52 = in_11 = in_10 = None
        tmp_54 = linear_3 + tmp_50;  linear_3 = tmp_50 = None
        tmp_55 = tmp_54 * in_9;  tmp_54 = in_9 = None
        tmp_56 = tmp_55 + in_8;  tmp_55 = in_8 = None
        linear_4 = torch.nn.functional.linear(tmp_56, in_13, in_12);  in_13 = in_12 = None
        tmp_58 = torch.nn.functional.relu(linear_4, inplace = False);  linear_4 = None
        linear_5 = torch.nn.functional.linear(tmp_58, in_17, in_16);  tmp_58 = in_17 = in_16 = None
        tmp_60 = linear_5 + tmp_56;  linear_5 = tmp_56 = None
        tmp_61 = tmp_60 * in_15;  tmp_60 = in_15 = None
        tmp_62 = tmp_61 + in_14;  tmp_61 = in_14 = None
        linear_6 = torch.nn.functional.linear(tmp_62, in_19, in_18);  in_19 = in_18 = None
        tmp_64 = torch.nn.functional.relu(linear_6, inplace = False);  linear_6 = None
        linear_7 = torch.nn.functional.linear(tmp_64, in_23, in_22);  tmp_64 = in_23 = in_22 = None
        tmp_66 = linear_7 + tmp_62;  linear_7 = tmp_62 = None
        tmp_67 = tmp_66 * in_21;  tmp_66 = in_21 = None
        tmp_68 = tmp_67 + in_20;  tmp_67 = in_20 = None
        linear_8 = torch.nn.functional.linear(tmp_68, in_25, in_24);  in_25 = in_24 = None
        tmp_70 = torch.nn.functional.relu(linear_8, inplace = False);  linear_8 = None
        linear_9 = torch.nn.functional.linear(tmp_70, in_33, in_32);  tmp_70 = in_33 = in_32 = None
        tmp_72 = linear_9 + tmp_68;  linear_9 = tmp_68 = None
        tmp_73 = tmp_72 * in_27;  tmp_72 = in_27 = None
        tmp_74 = tmp_73 + in_26;  tmp_73 = in_26 = None
        linear_10 = torch.nn.functional.linear(tmp_74, in_31, in_30);  tmp_74 = in_31 = in_30 = None
        tmp_76 = torch.nn.functional.dropout(linear_10, 0.0, False, False);  linear_10 = None
        tmp_77 = tmp_76 + in_37;  tmp_76 = in_37 = None
        tmp_78 = tmp_77 * in_29;  tmp_77 = in_29 = None
        tmp_79 = tmp_78 + in_28;  tmp_78 = in_28 = None
        tmp_80 = torch.tensor(1000);  tmp_80 = None
        tmp_81 = tmp_79[(slice(None, None, None), 0)]
        return (tmp_79, tmp_81)
        
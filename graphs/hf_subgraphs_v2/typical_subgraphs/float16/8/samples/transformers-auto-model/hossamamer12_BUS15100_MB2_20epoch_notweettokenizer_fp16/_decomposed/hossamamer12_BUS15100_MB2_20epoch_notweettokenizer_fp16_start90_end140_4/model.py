import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40):
        linear = torch.nn.functional.linear(in_39, in_5, in_4);  in_5 = in_4 = None
        tmp_37 = linear.view(128, -1, 4, 32);  linear = None
        tmp_38 = tmp_37.transpose(1, 2);  tmp_37 = None
        tmp_39 = in_37.transpose(-1, -2);  in_37 = None
        matmul = torch.matmul(in_40, tmp_39);  in_40 = tmp_39 = None
        tmp_41 = matmul / 5.656854249492381;  matmul = None
        tmp_42 = tmp_41 + in_36;  tmp_41 = in_36 = None
        tmp_43 = torch.nn.functional.softmax(tmp_42, dim = -1);  tmp_42 = None
        tmp_44 = torch.nn.functional.dropout(tmp_43, 0.1, False, False);  tmp_43 = None
        matmul_1 = torch.matmul(tmp_44, tmp_38);  tmp_44 = tmp_38 = None
        tmp_46 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_47 = tmp_46.contiguous();  tmp_46 = None
        tmp_48 = tmp_47.view((128, 64, 128));  tmp_47 = None
        linear_1 = torch.nn.functional.linear(tmp_48, in_3, in_2);  tmp_48 = in_3 = in_2 = None
        tmp_50 = linear_1 + in_38;  linear_1 = in_38 = None
        tmp_51 = tmp_50 * in_1;  tmp_50 = in_1 = None
        tmp_52 = tmp_51 + in_0;  tmp_51 = in_0 = None
        linear_2 = torch.nn.functional.linear(tmp_52, in_7, in_6);  in_7 = in_6 = None
        tmp_54 = torch.nn.functional.relu(linear_2, inplace = False);  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_54, in_11, in_10);  tmp_54 = in_11 = in_10 = None
        tmp_56 = linear_3 + tmp_52;  linear_3 = tmp_52 = None
        tmp_57 = tmp_56 * in_9;  tmp_56 = in_9 = None
        tmp_58 = tmp_57 + in_8;  tmp_57 = in_8 = None
        linear_4 = torch.nn.functional.linear(tmp_58, in_13, in_12);  in_13 = in_12 = None
        tmp_60 = torch.nn.functional.relu(linear_4, inplace = False);  linear_4 = None
        linear_5 = torch.nn.functional.linear(tmp_60, in_17, in_16);  tmp_60 = in_17 = in_16 = None
        tmp_62 = linear_5 + tmp_58;  linear_5 = tmp_58 = None
        tmp_63 = tmp_62 * in_15;  tmp_62 = in_15 = None
        tmp_64 = tmp_63 + in_14;  tmp_63 = in_14 = None
        linear_6 = torch.nn.functional.linear(tmp_64, in_19, in_18);  in_19 = in_18 = None
        tmp_66 = torch.nn.functional.relu(linear_6, inplace = False);  linear_6 = None
        linear_7 = torch.nn.functional.linear(tmp_66, in_23, in_22);  tmp_66 = in_23 = in_22 = None
        tmp_68 = linear_7 + tmp_64;  linear_7 = tmp_64 = None
        tmp_69 = tmp_68 * in_21;  tmp_68 = in_21 = None
        tmp_70 = tmp_69 + in_20;  tmp_69 = in_20 = None
        linear_8 = torch.nn.functional.linear(tmp_70, in_25, in_24);  in_25 = in_24 = None
        tmp_72 = torch.nn.functional.relu(linear_8, inplace = False);  linear_8 = None
        linear_9 = torch.nn.functional.linear(tmp_72, in_33, in_32);  tmp_72 = in_33 = in_32 = None
        tmp_74 = linear_9 + tmp_70;  linear_9 = tmp_70 = None
        tmp_75 = tmp_74 * in_27;  tmp_74 = in_27 = None
        tmp_76 = tmp_75 + in_26;  tmp_75 = in_26 = None
        linear_10 = torch.nn.functional.linear(tmp_76, in_31, in_30);  tmp_76 = in_31 = in_30 = None
        tmp_78 = torch.nn.functional.dropout(linear_10, 0.0, False, False);  linear_10 = None
        tmp_79 = tmp_78 + in_39;  tmp_78 = in_39 = None
        tmp_80 = tmp_79 * in_29;  tmp_79 = in_29 = None
        tmp_81 = tmp_80 + in_28;  tmp_80 = in_28 = None
        tmp_82 = torch.tensor(1000);  tmp_82 = None
        tmp_83 = tmp_81[(slice(None, None, None), 0)]
        linear_11 = torch.nn.functional.linear(tmp_83, in_35, in_34);  tmp_83 = in_35 = in_34 = None
        tmp_85 = torch.tanh(linear_11);  linear_11 = None
        return (tmp_81, tmp_85)
        
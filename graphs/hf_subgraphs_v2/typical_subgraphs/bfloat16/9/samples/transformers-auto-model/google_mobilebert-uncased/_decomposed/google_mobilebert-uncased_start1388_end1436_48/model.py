import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, in_0, in_1, in_2, in_3, in_4):
        linear = torch.nn.functional.linear(in_3, w_5, w_4);  w_5 = w_4 = None
        tmp_35 = linear.view(1, -1, 4, 32);  linear = None
        tmp_36 = tmp_35.transpose(1, 2);  tmp_35 = None
        tmp_37 = in_1.transpose(-1, -2);  in_1 = None
        matmul = torch.matmul(in_4, tmp_37);  in_4 = tmp_37 = None
        tmp_39 = matmul / 5.656854249492381;  matmul = None
        tmp_40 = tmp_39 + in_0;  tmp_39 = in_0 = None
        tmp_41 = torch.nn.functional.softmax(tmp_40, dim = -1);  tmp_40 = None
        tmp_42 = torch.nn.functional.dropout(tmp_41, 0.1, False, False);  tmp_41 = None
        matmul_1 = torch.matmul(tmp_42, tmp_36);  tmp_42 = tmp_36 = None
        tmp_44 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_45 = tmp_44.contiguous();  tmp_44 = None
        tmp_46 = tmp_45.view((1, 11, 128));  tmp_45 = None
        linear_1 = torch.nn.functional.linear(tmp_46, w_3, w_2);  tmp_46 = w_3 = w_2 = None
        tmp_48 = linear_1 + in_2;  linear_1 = in_2 = None
        tmp_49 = tmp_48 * w_1;  tmp_48 = w_1 = None
        tmp_50 = tmp_49 + w_0;  tmp_49 = w_0 = None
        linear_2 = torch.nn.functional.linear(tmp_50, w_7, w_6);  w_7 = w_6 = None
        tmp_52 = torch.nn.functional.relu(linear_2, inplace = False);  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_52, w_11, w_10);  tmp_52 = w_11 = w_10 = None
        tmp_54 = linear_3 + tmp_50;  linear_3 = tmp_50 = None
        tmp_55 = tmp_54 * w_9;  tmp_54 = w_9 = None
        tmp_56 = tmp_55 + w_8;  tmp_55 = w_8 = None
        linear_4 = torch.nn.functional.linear(tmp_56, w_13, w_12);  w_13 = w_12 = None
        tmp_58 = torch.nn.functional.relu(linear_4, inplace = False);  linear_4 = None
        linear_5 = torch.nn.functional.linear(tmp_58, w_17, w_16);  tmp_58 = w_17 = w_16 = None
        tmp_60 = linear_5 + tmp_56;  linear_5 = tmp_56 = None
        tmp_61 = tmp_60 * w_15;  tmp_60 = w_15 = None
        tmp_62 = tmp_61 + w_14;  tmp_61 = w_14 = None
        linear_6 = torch.nn.functional.linear(tmp_62, w_19, w_18);  w_19 = w_18 = None
        tmp_64 = torch.nn.functional.relu(linear_6, inplace = False);  linear_6 = None
        linear_7 = torch.nn.functional.linear(tmp_64, w_23, w_22);  tmp_64 = w_23 = w_22 = None
        tmp_66 = linear_7 + tmp_62;  linear_7 = tmp_62 = None
        tmp_67 = tmp_66 * w_21;  tmp_66 = w_21 = None
        tmp_68 = tmp_67 + w_20;  tmp_67 = w_20 = None
        linear_8 = torch.nn.functional.linear(tmp_68, w_25, w_24);  w_25 = w_24 = None
        tmp_70 = torch.nn.functional.relu(linear_8, inplace = False);  linear_8 = None
        linear_9 = torch.nn.functional.linear(tmp_70, w_33, w_32);  tmp_70 = w_33 = w_32 = None
        tmp_72 = linear_9 + tmp_68;  linear_9 = tmp_68 = None
        tmp_73 = tmp_72 * w_27;  tmp_72 = w_27 = None
        tmp_74 = tmp_73 + w_26;  tmp_73 = w_26 = None
        linear_10 = torch.nn.functional.linear(tmp_74, w_31, w_30);  tmp_74 = w_31 = w_30 = None
        tmp_76 = torch.nn.functional.dropout(linear_10, 0.0, False, False);  linear_10 = None
        tmp_77 = tmp_76 + in_3;  tmp_76 = in_3 = None
        tmp_78 = tmp_77 * w_29;  tmp_77 = w_29 = None
        tmp_79 = tmp_78 + w_28;  tmp_78 = w_28 = None
        tmp_80 = torch.tensor(1000);  tmp_80 = None
        tmp_81 = tmp_79[(slice(None, None, None), 0)]
        return (tmp_79, tmp_81)
        
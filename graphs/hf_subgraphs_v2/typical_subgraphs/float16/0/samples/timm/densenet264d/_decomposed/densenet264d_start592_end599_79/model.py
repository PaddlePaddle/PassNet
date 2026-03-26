import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70):
        tmp_6 = torch.nn.functional.relu(in_70, inplace = True);  in_70 = None
        to = tmp_6.to(torch.float16);  tmp_6 = None
        conv2d = torch.conv2d(to, in_0, None, (1, 1), (1, 1), (1, 1), 1);  to = in_0 = None
        tmp_8 = torch.cat([in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, conv2d], 1);  in_6 = in_7 = in_8 = in_9 = in_10 = in_11 = in_12 = in_13 = in_14 = in_15 = in_16 = in_17 = in_18 = in_19 = in_20 = in_21 = in_22 = in_23 = in_24 = in_25 = in_26 = in_27 = in_28 = in_29 = in_30 = in_31 = in_32 = in_33 = in_34 = in_35 = in_36 = in_37 = in_38 = in_39 = in_40 = in_41 = in_42 = in_43 = in_44 = in_45 = in_46 = in_47 = in_48 = in_49 = in_50 = in_51 = in_52 = in_53 = in_54 = in_55 = in_56 = in_57 = in_58 = in_59 = in_60 = in_61 = in_62 = in_63 = in_64 = in_65 = in_66 = in_67 = in_68 = in_69 = conv2d = None
        tmp_9 = torch.nn.functional.batch_norm(tmp_8, in_2, in_3, in_5, in_4, False, 0.1, 1e-05);  tmp_8 = in_2 = in_3 = in_5 = in_4 = None
        tmp_10 = torch.nn.functional.relu(tmp_9, inplace = True);  tmp_9 = None
        to_1 = tmp_10.to(torch.float16);  tmp_10 = None
        conv2d_1 = torch.conv2d(to_1, in_1, None, (1, 1), (0, 0), (1, 1), 1);  to_1 = in_1 = None
        tmp_12 = torch.nn.functional.avg_pool2d(conv2d_1, 2, 2, 0, False, True, None);  conv2d_1 = None
        return (tmp_12,)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61):
        tmp_5 = torch.nn.functional.relu(in_61, inplace = True);  in_61 = None
        conv2d = torch.conv2d(tmp_5, w_0, None, (1, 1), (1, 1), (1, 1), 1);  tmp_5 = w_0 = None
        tmp_7 = torch.cat([in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, conv2d], 1);  in_0 = in_1 = in_2 = in_3 = in_4 = in_5 = in_6 = in_7 = in_8 = in_9 = in_10 = in_11 = in_12 = in_13 = in_14 = in_15 = in_16 = in_17 = in_18 = in_19 = in_20 = in_21 = in_22 = in_23 = in_24 = in_25 = in_26 = in_27 = in_28 = in_29 = in_30 = in_31 = in_32 = in_33 = in_34 = in_35 = in_36 = in_37 = in_38 = in_39 = in_40 = in_41 = in_42 = in_43 = in_44 = in_45 = in_46 = in_47 = in_48 = in_49 = in_50 = in_51 = in_52 = in_53 = in_54 = in_55 = in_56 = in_57 = in_58 = in_59 = in_60 = None
        tmp_8 = torch.nn.functional.batch_norm(tmp_7, w_1, w_2, w_4, w_3, False, 0.1, 1e-05);  tmp_7 = w_1 = w_2 = w_4 = w_3 = None
        tmp_9 = torch.nn.functional.relu(tmp_8, inplace = True);  tmp_8 = None
        return (conv2d, tmp_9)
        
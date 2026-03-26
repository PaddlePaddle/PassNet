import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55):
        tmp_7 = torch.nn.functional.relu(in_55, inplace = True);  in_55 = None
        to = tmp_7.to(torch.bfloat16);  tmp_7 = None
        conv2d = torch.conv2d(to, in_2, None, (1, 1), (1, 1), (1, 1), 1);  to = in_2 = None
        tmp_9 = torch.cat([in_7, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, conv2d], 1);  in_7 = in_37 = in_38 = in_39 = in_40 = in_41 = in_42 = in_43 = in_44 = in_45 = in_46 = in_47 = in_48 = in_49 = in_50 = in_51 = in_52 = in_53 = in_54 = in_8 = in_9 = in_10 = in_11 = in_12 = in_13 = in_14 = in_15 = in_16 = in_17 = in_18 = in_19 = in_20 = in_21 = in_22 = in_23 = in_24 = in_25 = in_26 = in_27 = in_28 = in_29 = in_30 = in_31 = in_32 = in_33 = in_34 = in_35 = in_36 = conv2d = None
        tmp_10 = torch.nn.functional.batch_norm(tmp_9, in_3, in_4, in_6, in_5, False, 0.1, 1e-05);  tmp_9 = in_3 = in_4 = in_6 = in_5 = None
        tmp_11 = torch.nn.functional.relu(tmp_10, inplace = True);  tmp_10 = None
        tmp_12 = torch.nn.functional.adaptive_avg_pool2d(tmp_11, 1);  tmp_11 = None
        tmp_13 = tmp_12.flatten(1, -1);  tmp_12 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False);  tmp_13 = None
        to_1 = tmp_14.to(torch.bfloat16);  tmp_14 = None
        linear = torch.nn.functional.linear(to_1, in_1, in_0);  to_1 = in_1 = in_0 = None
        return (linear,)
        
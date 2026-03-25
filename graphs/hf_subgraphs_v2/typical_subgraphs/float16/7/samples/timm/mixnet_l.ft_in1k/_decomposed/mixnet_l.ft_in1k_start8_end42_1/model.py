import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26):
        tmp_25 = in_26 + in_25;  in_26 = in_25 = None
        split = torch.functional.split(tmp_25, [16, 16], 1);  tmp_25 = None
        tmp_27 = split[0]
        tmp_28 = split[1];  split = None
        conv2d = torch.conv2d(tmp_27, in_15, None, (1, 1), (0, 0), (1, 1), 1);  tmp_27 = in_15 = None
        conv2d_1 = torch.conv2d(tmp_28, in_16, None, (1, 1), (0, 0), (1, 1), 1);  tmp_28 = in_16 = None
        tmp_31 = torch.cat([conv2d, conv2d_1], 1);  conv2d = conv2d_1 = None
        tmp_32 = torch.nn.functional.batch_norm(tmp_31, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_31 = in_0 = in_1 = in_3 = in_2 = None
        tmp_33 = torch.nn.functional.relu(tmp_32, inplace = True);  tmp_32 = None
        split_1 = torch.functional.split(tmp_33, [64, 64, 64], 1);  tmp_33 = None
        tmp_35 = split_1[0]
        tmp_36 = split_1[1]
        tmp_37 = split_1[2];  split_1 = None
        conv2d_2 = torch.conv2d(tmp_35, in_12, None, (2, 2), (1, 1), (1, 1), 64);  tmp_35 = in_12 = None
        conv2d_3 = torch.conv2d(tmp_36, in_13, None, (2, 2), (2, 2), (1, 1), 64);  tmp_36 = in_13 = None
        conv2d_4 = torch.conv2d(tmp_37, in_14, None, (2, 2), (3, 3), (1, 1), 64);  tmp_37 = in_14 = None
        tmp_41 = torch.cat([conv2d_2, conv2d_3, conv2d_4], 1);  conv2d_2 = conv2d_3 = conv2d_4 = None
        tmp_42 = torch.nn.functional.batch_norm(tmp_41, in_4, in_5, in_7, in_6, False, 0.1, 1e-05);  tmp_41 = in_4 = in_5 = in_7 = in_6 = None
        tmp_43 = torch.nn.functional.relu(tmp_42, inplace = True);  tmp_42 = None
        split_2 = torch.functional.split(tmp_43, [96, 96], 1);  tmp_43 = None
        tmp_45 = split_2[0]
        tmp_46 = split_2[1];  split_2 = None
        conv2d_5 = torch.conv2d(tmp_45, in_17, None, (1, 1), (0, 0), (1, 1), 1);  tmp_45 = in_17 = None
        conv2d_6 = torch.conv2d(tmp_46, in_18, None, (1, 1), (0, 0), (1, 1), 1);  tmp_46 = in_18 = None
        tmp_49 = torch.cat([conv2d_5, conv2d_6], 1);  conv2d_5 = conv2d_6 = None
        tmp_50 = torch.nn.functional.batch_norm(tmp_49, in_8, in_9, in_11, in_10, False, 0.1, 1e-05);  tmp_49 = in_8 = in_9 = in_11 = in_10 = None
        split_3 = torch.functional.split(tmp_50, [20, 20], 1)
        tmp_52 = split_3[0]
        tmp_53 = split_3[1];  split_3 = None
        conv2d_7 = torch.conv2d(tmp_52, in_23, None, (1, 1), (0, 0), (1, 1), 1);  tmp_52 = in_23 = None
        conv2d_8 = torch.conv2d(tmp_53, in_24, None, (1, 1), (0, 0), (1, 1), 1);  tmp_53 = in_24 = None
        tmp_56 = torch.cat([conv2d_7, conv2d_8], 1);  conv2d_7 = conv2d_8 = None
        tmp_57 = torch.nn.functional.batch_norm(tmp_56, in_19, in_20, in_22, in_21, False, 0.1, 1e-05);  tmp_56 = in_19 = in_20 = in_22 = in_21 = None
        tmp_58 = torch.nn.functional.relu(tmp_57, inplace = True);  tmp_57 = None
        return (tmp_50, tmp_58)
        
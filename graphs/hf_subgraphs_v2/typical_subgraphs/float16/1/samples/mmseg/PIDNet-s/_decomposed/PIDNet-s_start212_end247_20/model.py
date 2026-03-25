import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36):
        in_35 += in_36;  in_37 = in_35;  in_35 = in_36 = None
        tmp_35 = torch.nn.functional.batch_norm(in_34, in_9, in_10, in_12, in_11, False, 0.1, 1e-05);  in_9 = in_10 = in_12 = in_11 = None
        tmp_36 = torch.nn.functional.relu(tmp_35, inplace = True);  tmp_35 = None
        conv2d = torch.conv2d(tmp_36, in_13, None, (1, 1), (0, 0), (1, 1), 1);  tmp_36 = in_13 = None
        tmp_38 = torch.nn.functional.avg_pool2d(in_34, 5, 2, 2, False, True, None)
        tmp_39 = torch.nn.functional.batch_norm(tmp_38, in_14, in_15, in_17, in_16, False, 0.1, 1e-05);  tmp_38 = in_14 = in_15 = in_17 = in_16 = None
        tmp_40 = torch.nn.functional.relu(tmp_39, inplace = True);  tmp_39 = None
        conv2d_1 = torch.conv2d(tmp_40, in_18, None, (1, 1), (0, 0), (1, 1), 1);  tmp_40 = in_18 = None
        tmp_42 = torch.nn.functional.interpolate(conv2d_1, size = (8, 8), mode = 'bilinear', align_corners = False);  conv2d_1 = None
        tmp_43 = tmp_42 + conv2d;  tmp_42 = None
        tmp_44 = torch.nn.functional.avg_pool2d(in_34, 9, 4, 4, False, True, None)
        tmp_45 = torch.nn.functional.batch_norm(tmp_44, in_19, in_20, in_22, in_21, False, 0.1, 1e-05);  tmp_44 = in_19 = in_20 = in_22 = in_21 = None
        tmp_46 = torch.nn.functional.relu(tmp_45, inplace = True);  tmp_45 = None
        conv2d_2 = torch.conv2d(tmp_46, in_23, None, (1, 1), (0, 0), (1, 1), 1);  tmp_46 = in_23 = None
        tmp_48 = torch.nn.functional.interpolate(conv2d_2, size = (8, 8), mode = 'bilinear', align_corners = False);  conv2d_2 = None
        tmp_49 = tmp_48 + conv2d;  tmp_48 = None
        tmp_50 = torch.nn.functional.avg_pool2d(in_34, 17, 8, 8, False, True, None)
        tmp_51 = torch.nn.functional.batch_norm(tmp_50, in_24, in_25, in_27, in_26, False, 0.1, 1e-05);  tmp_50 = in_24 = in_25 = in_27 = in_26 = None
        tmp_52 = torch.nn.functional.relu(tmp_51, inplace = True);  tmp_51 = None
        conv2d_3 = torch.conv2d(tmp_52, in_28, None, (1, 1), (0, 0), (1, 1), 1);  tmp_52 = in_28 = None
        tmp_54 = torch.nn.functional.interpolate(conv2d_3, size = (8, 8), mode = 'bilinear', align_corners = False);  conv2d_3 = None
        tmp_55 = tmp_54 + conv2d;  tmp_54 = None
        tmp_56 = torch.nn.functional.adaptive_avg_pool2d(in_34, (1, 1));  in_34 = None
        tmp_57 = torch.nn.functional.batch_norm(tmp_56, in_29, in_30, in_32, in_31, False, 0.1, 1e-05);  tmp_56 = in_29 = in_30 = in_32 = in_31 = None
        tmp_58 = torch.nn.functional.relu(tmp_57, inplace = True);  tmp_57 = None
        conv2d_4 = torch.conv2d(tmp_58, in_33, None, (1, 1), (0, 0), (1, 1), 1);  tmp_58 = in_33 = None
        tmp_60 = torch.nn.functional.interpolate(conv2d_4, size = (8, 8), mode = 'bilinear', align_corners = False);  conv2d_4 = None
        tmp_61 = tmp_60 + conv2d;  tmp_60 = None
        tmp_62 = torch.cat([tmp_43, tmp_49, tmp_55, tmp_61], dim = 1);  tmp_43 = tmp_49 = tmp_55 = tmp_61 = None
        tmp_63 = torch.nn.functional.batch_norm(tmp_62, in_4, in_5, in_7, in_6, False, 0.1, 1e-05);  tmp_62 = in_4 = in_5 = in_7 = in_6 = None
        tmp_64 = torch.nn.functional.relu(tmp_63, inplace = True);  tmp_63 = None
        conv2d_5 = torch.conv2d(tmp_64, in_8, None, (1, 1), (1, 1), (1, 1), 4);  tmp_64 = in_8 = None
        tmp_66 = torch.cat([conv2d, conv2d_5], dim = 1);  conv2d = conv2d_5 = None
        tmp_67 = torch.nn.functional.batch_norm(tmp_66, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_66 = in_0 = in_1 = in_3 = in_2 = None
        tmp_68 = torch.nn.functional.relu(tmp_67, inplace = True);  tmp_67 = None
        return (in_37, tmp_68)
        
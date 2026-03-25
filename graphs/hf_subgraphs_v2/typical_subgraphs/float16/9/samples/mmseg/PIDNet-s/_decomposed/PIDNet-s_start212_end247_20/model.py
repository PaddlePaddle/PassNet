import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, in_0, in_1, in_2):
        in_1 += in_2;  in_3 = in_1;  in_1 = in_2 = None
        tmp_35 = torch.nn.functional.batch_norm(in_0, w_9, w_10, w_12, w_11, False, 0.1, 1e-05);  w_9 = w_10 = w_12 = w_11 = None
        tmp_36 = torch.nn.functional.relu(tmp_35, inplace = True);  tmp_35 = None
        conv2d = torch.conv2d(tmp_36, w_13, None, (1, 1), (0, 0), (1, 1), 1);  tmp_36 = w_13 = None
        tmp_38 = torch.nn.functional.avg_pool2d(in_0, 5, 2, 2, False, True, None)
        tmp_39 = torch.nn.functional.batch_norm(tmp_38, w_14, w_15, w_17, w_16, False, 0.1, 1e-05);  tmp_38 = w_14 = w_15 = w_17 = w_16 = None
        tmp_40 = torch.nn.functional.relu(tmp_39, inplace = True);  tmp_39 = None
        conv2d_1 = torch.conv2d(tmp_40, w_18, None, (1, 1), (0, 0), (1, 1), 1);  tmp_40 = w_18 = None
        tmp_42 = torch.nn.functional.interpolate(conv2d_1, size = (8, 8), mode = 'bilinear', align_corners = False);  conv2d_1 = None
        tmp_43 = tmp_42 + conv2d;  tmp_42 = None
        tmp_44 = torch.nn.functional.avg_pool2d(in_0, 9, 4, 4, False, True, None)
        tmp_45 = torch.nn.functional.batch_norm(tmp_44, w_19, w_20, w_22, w_21, False, 0.1, 1e-05);  tmp_44 = w_19 = w_20 = w_22 = w_21 = None
        tmp_46 = torch.nn.functional.relu(tmp_45, inplace = True);  tmp_45 = None
        conv2d_2 = torch.conv2d(tmp_46, w_23, None, (1, 1), (0, 0), (1, 1), 1);  tmp_46 = w_23 = None
        tmp_48 = torch.nn.functional.interpolate(conv2d_2, size = (8, 8), mode = 'bilinear', align_corners = False);  conv2d_2 = None
        tmp_49 = tmp_48 + conv2d;  tmp_48 = None
        tmp_50 = torch.nn.functional.avg_pool2d(in_0, 17, 8, 8, False, True, None)
        tmp_51 = torch.nn.functional.batch_norm(tmp_50, w_24, w_25, w_27, w_26, False, 0.1, 1e-05);  tmp_50 = w_24 = w_25 = w_27 = w_26 = None
        tmp_52 = torch.nn.functional.relu(tmp_51, inplace = True);  tmp_51 = None
        conv2d_3 = torch.conv2d(tmp_52, w_28, None, (1, 1), (0, 0), (1, 1), 1);  tmp_52 = w_28 = None
        tmp_54 = torch.nn.functional.interpolate(conv2d_3, size = (8, 8), mode = 'bilinear', align_corners = False);  conv2d_3 = None
        tmp_55 = tmp_54 + conv2d;  tmp_54 = None
        tmp_56 = torch.nn.functional.adaptive_avg_pool2d(in_0, (1, 1));  in_0 = None
        tmp_57 = torch.nn.functional.batch_norm(tmp_56, w_29, w_30, w_32, w_31, False, 0.1, 1e-05);  tmp_56 = w_29 = w_30 = w_32 = w_31 = None
        tmp_58 = torch.nn.functional.relu(tmp_57, inplace = True);  tmp_57 = None
        conv2d_4 = torch.conv2d(tmp_58, w_33, None, (1, 1), (0, 0), (1, 1), 1);  tmp_58 = w_33 = None
        tmp_60 = torch.nn.functional.interpolate(conv2d_4, size = (8, 8), mode = 'bilinear', align_corners = False);  conv2d_4 = None
        tmp_61 = tmp_60 + conv2d;  tmp_60 = None
        tmp_62 = torch.cat([tmp_43, tmp_49, tmp_55, tmp_61], dim = 1);  tmp_43 = tmp_49 = tmp_55 = tmp_61 = None
        tmp_63 = torch.nn.functional.batch_norm(tmp_62, w_4, w_5, w_7, w_6, False, 0.1, 1e-05);  tmp_62 = w_4 = w_5 = w_7 = w_6 = None
        tmp_64 = torch.nn.functional.relu(tmp_63, inplace = True);  tmp_63 = None
        conv2d_5 = torch.conv2d(tmp_64, w_8, None, (1, 1), (1, 1), (1, 1), 4);  tmp_64 = w_8 = None
        tmp_66 = torch.cat([conv2d, conv2d_5], dim = 1);  conv2d = conv2d_5 = None
        tmp_67 = torch.nn.functional.batch_norm(tmp_66, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  tmp_66 = w_0 = w_1 = w_3 = w_2 = None
        tmp_68 = torch.nn.functional.relu(tmp_67, inplace = True);  tmp_67 = None
        return (in_3, tmp_68)
        
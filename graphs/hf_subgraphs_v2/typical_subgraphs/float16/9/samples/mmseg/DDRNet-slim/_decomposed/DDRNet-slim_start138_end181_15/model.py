import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, w_34, w_35, w_36, w_37, w_38, w_39, w_40, w_41, w_42, w_43, w_44, w_45, w_46, w_47, w_48, in_0, in_1):
        in_1 += in_0;  in_2 = in_1;  in_1 = in_0 = None
        tmp_50 = torch.nn.functional.batch_norm(in_2, w_24, w_25, w_27, w_26, False, 0.1, 1e-05);  w_24 = w_25 = w_27 = w_26 = None
        tmp_51 = torch.nn.functional.relu(tmp_50, inplace = True);  tmp_50 = None
        conv2d = torch.conv2d(tmp_51, w_28, None, (1, 1), (0, 0), (1, 1), 1);  tmp_51 = w_28 = None
        tmp_53 = torch.nn.functional.avg_pool2d(in_2, 5, 2, 2, False, True, None)
        tmp_54 = torch.nn.functional.batch_norm(tmp_53, w_29, w_30, w_32, w_31, False, 0.1, 1e-05);  tmp_53 = w_29 = w_30 = w_32 = w_31 = None
        tmp_55 = torch.nn.functional.relu(tmp_54, inplace = True);  tmp_54 = None
        conv2d_1 = torch.conv2d(tmp_55, w_33, None, (1, 1), (0, 0), (1, 1), 1);  tmp_55 = w_33 = None
        tmp_57 = torch.nn.functional.interpolate(conv2d_1, size = (8, 8), mode = 'bilinear');  conv2d_1 = None
        tmp_58 = tmp_57 + conv2d;  tmp_57 = None
        tmp_59 = torch.nn.functional.batch_norm(tmp_58, w_4, w_5, w_7, w_6, False, 0.1, 1e-05);  tmp_58 = w_4 = w_5 = w_7 = w_6 = None
        tmp_60 = torch.nn.functional.relu(tmp_59, inplace = True);  tmp_59 = None
        conv2d_2 = torch.conv2d(tmp_60, w_8, None, (1, 1), (1, 1), (1, 1), 1);  tmp_60 = w_8 = None
        tmp_62 = torch.nn.functional.avg_pool2d(in_2, 9, 4, 4, False, True, None)
        tmp_63 = torch.nn.functional.batch_norm(tmp_62, w_34, w_35, w_37, w_36, False, 0.1, 1e-05);  tmp_62 = w_34 = w_35 = w_37 = w_36 = None
        tmp_64 = torch.nn.functional.relu(tmp_63, inplace = True);  tmp_63 = None
        conv2d_3 = torch.conv2d(tmp_64, w_38, None, (1, 1), (0, 0), (1, 1), 1);  tmp_64 = w_38 = None
        tmp_66 = torch.nn.functional.interpolate(conv2d_3, size = (8, 8), mode = 'bilinear');  conv2d_3 = None
        tmp_67 = tmp_66 + conv2d_2;  tmp_66 = None
        tmp_68 = torch.nn.functional.batch_norm(tmp_67, w_9, w_10, w_12, w_11, False, 0.1, 1e-05);  tmp_67 = w_9 = w_10 = w_12 = w_11 = None
        tmp_69 = torch.nn.functional.relu(tmp_68, inplace = True);  tmp_68 = None
        conv2d_4 = torch.conv2d(tmp_69, w_13, None, (1, 1), (1, 1), (1, 1), 1);  tmp_69 = w_13 = None
        tmp_71 = torch.nn.functional.avg_pool2d(in_2, 17, 8, 8, False, True, None)
        tmp_72 = torch.nn.functional.batch_norm(tmp_71, w_39, w_40, w_42, w_41, False, 0.1, 1e-05);  tmp_71 = w_39 = w_40 = w_42 = w_41 = None
        tmp_73 = torch.nn.functional.relu(tmp_72, inplace = True);  tmp_72 = None
        conv2d_5 = torch.conv2d(tmp_73, w_43, None, (1, 1), (0, 0), (1, 1), 1);  tmp_73 = w_43 = None
        tmp_75 = torch.nn.functional.interpolate(conv2d_5, size = (8, 8), mode = 'bilinear');  conv2d_5 = None
        tmp_76 = tmp_75 + conv2d_4;  tmp_75 = None
        tmp_77 = torch.nn.functional.batch_norm(tmp_76, w_14, w_15, w_17, w_16, False, 0.1, 1e-05);  tmp_76 = w_14 = w_15 = w_17 = w_16 = None
        tmp_78 = torch.nn.functional.relu(tmp_77, inplace = True);  tmp_77 = None
        conv2d_6 = torch.conv2d(tmp_78, w_18, None, (1, 1), (1, 1), (1, 1), 1);  tmp_78 = w_18 = None
        tmp_80 = torch.nn.functional.adaptive_avg_pool2d(in_2, (1, 1))
        tmp_81 = torch.nn.functional.batch_norm(tmp_80, w_44, w_45, w_47, w_46, False, 0.1, 1e-05);  tmp_80 = w_44 = w_45 = w_47 = w_46 = None
        tmp_82 = torch.nn.functional.relu(tmp_81, inplace = True);  tmp_81 = None
        conv2d_7 = torch.conv2d(tmp_82, w_48, None, (1, 1), (0, 0), (1, 1), 1);  tmp_82 = w_48 = None
        tmp_84 = torch.nn.functional.interpolate(conv2d_7, size = (8, 8), mode = 'bilinear');  conv2d_7 = None
        tmp_85 = tmp_84 + conv2d_6;  tmp_84 = None
        tmp_86 = torch.nn.functional.batch_norm(tmp_85, w_19, w_20, w_22, w_21, False, 0.1, 1e-05);  tmp_85 = w_19 = w_20 = w_22 = w_21 = None
        tmp_87 = torch.nn.functional.relu(tmp_86, inplace = True);  tmp_86 = None
        conv2d_8 = torch.conv2d(tmp_87, w_23, None, (1, 1), (1, 1), (1, 1), 1);  tmp_87 = w_23 = None
        tmp_89 = torch.cat([conv2d, conv2d_2, conv2d_4, conv2d_6, conv2d_8], dim = 1);  conv2d = conv2d_2 = conv2d_4 = conv2d_6 = conv2d_8 = None
        tmp_90 = torch.nn.functional.batch_norm(tmp_89, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  tmp_89 = w_0 = w_1 = w_3 = w_2 = None
        tmp_91 = torch.nn.functional.relu(tmp_90, inplace = True);  tmp_90 = None
        return (in_2, tmp_91)
        
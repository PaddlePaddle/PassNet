import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50):
        in_50 += in_49;  in_51 = in_50;  in_50 = in_49 = None
        tmp_50 = torch.nn.functional.batch_norm(in_51, in_24, in_25, in_27, in_26, False, 0.1, 1e-05);  in_24 = in_25 = in_27 = in_26 = None
        tmp_51 = torch.nn.functional.relu(tmp_50, inplace = True);  tmp_50 = None
        conv2d = torch.conv2d(tmp_51, in_28, None, (1, 1), (0, 0), (1, 1), 1);  tmp_51 = in_28 = None
        tmp_53 = torch.nn.functional.avg_pool2d(in_51, 5, 2, 2, False, True, None)
        tmp_54 = torch.nn.functional.batch_norm(tmp_53, in_29, in_30, in_32, in_31, False, 0.1, 1e-05);  tmp_53 = in_29 = in_30 = in_32 = in_31 = None
        tmp_55 = torch.nn.functional.relu(tmp_54, inplace = True);  tmp_54 = None
        conv2d_1 = torch.conv2d(tmp_55, in_33, None, (1, 1), (0, 0), (1, 1), 1);  tmp_55 = in_33 = None
        tmp_57 = torch.nn.functional.interpolate(conv2d_1, size = (8, 8), mode = 'bilinear');  conv2d_1 = None
        tmp_58 = tmp_57 + conv2d;  tmp_57 = None
        tmp_59 = torch.nn.functional.batch_norm(tmp_58, in_4, in_5, in_7, in_6, False, 0.1, 1e-05);  tmp_58 = in_4 = in_5 = in_7 = in_6 = None
        tmp_60 = torch.nn.functional.relu(tmp_59, inplace = True);  tmp_59 = None
        conv2d_2 = torch.conv2d(tmp_60, in_8, None, (1, 1), (1, 1), (1, 1), 1);  tmp_60 = in_8 = None
        tmp_62 = torch.nn.functional.avg_pool2d(in_51, 9, 4, 4, False, True, None)
        tmp_63 = torch.nn.functional.batch_norm(tmp_62, in_34, in_35, in_37, in_36, False, 0.1, 1e-05);  tmp_62 = in_34 = in_35 = in_37 = in_36 = None
        tmp_64 = torch.nn.functional.relu(tmp_63, inplace = True);  tmp_63 = None
        conv2d_3 = torch.conv2d(tmp_64, in_38, None, (1, 1), (0, 0), (1, 1), 1);  tmp_64 = in_38 = None
        tmp_66 = torch.nn.functional.interpolate(conv2d_3, size = (8, 8), mode = 'bilinear');  conv2d_3 = None
        tmp_67 = tmp_66 + conv2d_2;  tmp_66 = None
        tmp_68 = torch.nn.functional.batch_norm(tmp_67, in_9, in_10, in_12, in_11, False, 0.1, 1e-05);  tmp_67 = in_9 = in_10 = in_12 = in_11 = None
        tmp_69 = torch.nn.functional.relu(tmp_68, inplace = True);  tmp_68 = None
        conv2d_4 = torch.conv2d(tmp_69, in_13, None, (1, 1), (1, 1), (1, 1), 1);  tmp_69 = in_13 = None
        tmp_71 = torch.nn.functional.avg_pool2d(in_51, 17, 8, 8, False, True, None)
        tmp_72 = torch.nn.functional.batch_norm(tmp_71, in_39, in_40, in_42, in_41, False, 0.1, 1e-05);  tmp_71 = in_39 = in_40 = in_42 = in_41 = None
        tmp_73 = torch.nn.functional.relu(tmp_72, inplace = True);  tmp_72 = None
        conv2d_5 = torch.conv2d(tmp_73, in_43, None, (1, 1), (0, 0), (1, 1), 1);  tmp_73 = in_43 = None
        tmp_75 = torch.nn.functional.interpolate(conv2d_5, size = (8, 8), mode = 'bilinear');  conv2d_5 = None
        tmp_76 = tmp_75 + conv2d_4;  tmp_75 = None
        tmp_77 = torch.nn.functional.batch_norm(tmp_76, in_14, in_15, in_17, in_16, False, 0.1, 1e-05);  tmp_76 = in_14 = in_15 = in_17 = in_16 = None
        tmp_78 = torch.nn.functional.relu(tmp_77, inplace = True);  tmp_77 = None
        conv2d_6 = torch.conv2d(tmp_78, in_18, None, (1, 1), (1, 1), (1, 1), 1);  tmp_78 = in_18 = None
        tmp_80 = torch.nn.functional.adaptive_avg_pool2d(in_51, (1, 1))
        tmp_81 = torch.nn.functional.batch_norm(tmp_80, in_44, in_45, in_47, in_46, False, 0.1, 1e-05);  tmp_80 = in_44 = in_45 = in_47 = in_46 = None
        tmp_82 = torch.nn.functional.relu(tmp_81, inplace = True);  tmp_81 = None
        conv2d_7 = torch.conv2d(tmp_82, in_48, None, (1, 1), (0, 0), (1, 1), 1);  tmp_82 = in_48 = None
        tmp_84 = torch.nn.functional.interpolate(conv2d_7, size = (8, 8), mode = 'bilinear');  conv2d_7 = None
        tmp_85 = tmp_84 + conv2d_6;  tmp_84 = None
        tmp_86 = torch.nn.functional.batch_norm(tmp_85, in_19, in_20, in_22, in_21, False, 0.1, 1e-05);  tmp_85 = in_19 = in_20 = in_22 = in_21 = None
        tmp_87 = torch.nn.functional.relu(tmp_86, inplace = True);  tmp_86 = None
        conv2d_8 = torch.conv2d(tmp_87, in_23, None, (1, 1), (1, 1), (1, 1), 1);  tmp_87 = in_23 = None
        tmp_89 = torch.cat([conv2d, conv2d_2, conv2d_4, conv2d_6, conv2d_8], dim = 1);  conv2d = conv2d_2 = conv2d_4 = conv2d_6 = conv2d_8 = None
        tmp_90 = torch.nn.functional.batch_norm(tmp_89, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_89 = in_0 = in_1 = in_3 = in_2 = None
        tmp_91 = torch.nn.functional.relu(tmp_90, inplace = True);  tmp_90 = None
        return (in_51, tmp_91)
        
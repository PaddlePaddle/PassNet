import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor, in_29 : torch.Tensor, in_30 : torch.Tensor, in_31 : torch.Tensor, in_32 : torch.Tensor, in_33 : torch.Tensor, in_34 : torch.Tensor, in_35 : torch.Tensor, in_36 : torch.Tensor, in_37 : torch.Tensor, in_38 : torch.Tensor, in_39 : torch.Tensor, in_40 : torch.Tensor, in_41 : torch.Tensor, in_42 : torch.Tensor, in_43 : torch.Tensor, in_44 : torch.Tensor, in_45 : torch.Tensor, in_46 : torch.Tensor, in_47 : torch.Tensor, in_48 : torch.Tensor, in_49 : torch.Tensor, in_50 : torch.Tensor, in_51 : torch.Tensor, in_52 : torch.Tensor, in_53 : torch.Tensor, in_54 : torch.Tensor, in_55 : torch.Tensor, in_56 : torch.Tensor, in_57 : torch.Tensor, in_58 : torch.Tensor, in_59 : torch.Tensor, in_60 : torch.Tensor, in_61 : torch.Tensor, in_62 : torch.Tensor, in_63 : torch.Tensor, in_64 : torch.Tensor, in_65 : torch.Tensor, in_66 : torch.Tensor, in_67 : torch.Tensor, in_68 : torch.Tensor, in_69 : torch.Tensor, in_70 : torch.Tensor, in_71 : torch.Tensor, in_72 : torch.Tensor, in_73 : torch.Tensor, in_74 : torch.Tensor, in_75 : torch.Tensor):
        conv2d = torch.conv2d(in_75, in_74, None, (2, 2), (1, 1), (1, 1), 1);  in_75 = in_74 = None
        tmp_77 = torch.nn.functional.batch_norm(conv2d, in_63, in_64, in_66, in_65, False, 0.1, 1e-05);  conv2d = in_63 = in_64 = in_66 = in_65 = None
        tmp_78 = torch.nn.functional.silu(tmp_77, inplace = True);  tmp_77 = None
        conv2d_1 = torch.conv2d(tmp_78, in_4, None, (1, 1), (1, 1), (1, 1), 1);  tmp_78 = in_4 = None
        tmp_80 = torch.nn.functional.batch_norm(conv2d_1, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  conv2d_1 = in_0 = in_1 = in_3 = in_2 = None
        tmp_81 = torch.nn.functional.silu(tmp_80, inplace = True);  tmp_80 = None
        conv2d_2 = torch.conv2d(tmp_81, in_13, None, (2, 2), (1, 1), (1, 1), 1);  tmp_81 = in_13 = None
        tmp_83 = torch.nn.functional.batch_norm(conv2d_2, in_5, in_6, in_8, in_7, False, 0.1, 1e-05);  conv2d_2 = in_5 = in_6 = in_8 = in_7 = None
        tmp_84 = torch.nn.functional.silu(tmp_83, inplace = True);  tmp_83 = None
        conv2d_3 = torch.conv2d(tmp_84, in_14, None, (1, 1), (0, 0), (1, 1), 1);  tmp_84 = in_14 = None
        tmp_86 = torch.nn.functional.batch_norm(conv2d_3, in_9, in_10, in_12, in_11, False, 0.1, 1e-05);  conv2d_3 = in_9 = in_10 = in_12 = in_11 = None
        conv2d_4 = torch.conv2d(tmp_86, in_23, None, (2, 2), (1, 1), (1, 1), 1);  tmp_86 = in_23 = None
        tmp_88 = torch.nn.functional.batch_norm(conv2d_4, in_15, in_16, in_18, in_17, False, 0.1, 1e-05);  conv2d_4 = in_15 = in_16 = in_18 = in_17 = None
        tmp_89 = torch.nn.functional.silu(tmp_88, inplace = True);  tmp_88 = None
        conv2d_5 = torch.conv2d(tmp_89, in_24, None, (1, 1), (0, 0), (1, 1), 1);  tmp_89 = in_24 = None
        tmp_91 = torch.nn.functional.batch_norm(conv2d_5, in_19, in_20, in_22, in_21, False, 0.1, 1e-05);  conv2d_5 = in_19 = in_20 = in_22 = in_21 = None
        conv2d_6 = torch.conv2d(tmp_91, in_38, None, (1, 1), (0, 0), (1, 1), 1);  tmp_91 = in_38 = None
        tmp_93 = torch.nn.functional.batch_norm(conv2d_6, in_25, in_26, in_28, in_27, False, 0.1, 1e-05);  conv2d_6 = in_25 = in_26 = in_28 = in_27 = None
        tmp_94 = torch.nn.functional.silu(tmp_93, inplace = True);  tmp_93 = None
        conv2d_7 = torch.conv2d(tmp_94, in_37, None, (2, 2), (1, 1), (1, 1), 128);  tmp_94 = in_37 = None
        tmp_96 = torch.nn.functional.batch_norm(conv2d_7, in_29, in_30, in_32, in_31, False, 0.1, 1e-05);  conv2d_7 = in_29 = in_30 = in_32 = in_31 = None
        tmp_97 = torch.nn.functional.silu(tmp_96, inplace = True);  tmp_96 = None
        tmp_98 = tmp_97.mean((2, 3), keepdim = True)
        conv2d_8 = torch.conv2d(tmp_98, in_43, in_42, (1, 1), (0, 0), (1, 1), 1);  tmp_98 = in_43 = in_42 = None
        tmp_100 = torch.nn.functional.silu(conv2d_8, inplace = True);  conv2d_8 = None
        conv2d_9 = torch.conv2d(tmp_100, in_41, in_40, (1, 1), (0, 0), (1, 1), 1);  tmp_100 = in_41 = in_40 = None
        tmp_102 = torch.sigmoid(conv2d_9);  conv2d_9 = None
        tmp_103 = tmp_97 * tmp_102;  tmp_97 = tmp_102 = None
        conv2d_10 = torch.conv2d(tmp_103, in_39, None, (1, 1), (0, 0), (1, 1), 1);  tmp_103 = in_39 = None
        tmp_105 = torch.nn.functional.batch_norm(conv2d_10, in_33, in_34, in_36, in_35, False, 0.1, 1e-05);  conv2d_10 = in_33 = in_34 = in_36 = in_35 = None
        conv2d_11 = torch.conv2d(tmp_105, in_57, None, (1, 1), (0, 0), (1, 1), 1);  tmp_105 = in_57 = None
        tmp_107 = torch.nn.functional.batch_norm(conv2d_11, in_44, in_45, in_47, in_46, False, 0.1, 1e-05);  conv2d_11 = in_44 = in_45 = in_47 = in_46 = None
        tmp_108 = torch.nn.functional.silu(tmp_107, inplace = True);  tmp_107 = None
        conv2d_12 = torch.conv2d(tmp_108, in_56, None, (2, 2), (1, 1), (1, 1), 192);  tmp_108 = in_56 = None
        tmp_110 = torch.nn.functional.batch_norm(conv2d_12, in_48, in_49, in_51, in_50, False, 0.1, 1e-05);  conv2d_12 = in_48 = in_49 = in_51 = in_50 = None
        tmp_111 = torch.nn.functional.silu(tmp_110, inplace = True);  tmp_110 = None
        tmp_112 = tmp_111.mean((2, 3), keepdim = True)
        conv2d_13 = torch.conv2d(tmp_112, in_62, in_61, (1, 1), (0, 0), (1, 1), 1);  tmp_112 = in_62 = in_61 = None
        tmp_114 = torch.nn.functional.silu(conv2d_13, inplace = True);  conv2d_13 = None
        conv2d_14 = torch.conv2d(tmp_114, in_60, in_59, (1, 1), (0, 0), (1, 1), 1);  tmp_114 = in_60 = in_59 = None
        tmp_116 = torch.sigmoid(conv2d_14);  conv2d_14 = None
        tmp_117 = tmp_111 * tmp_116;  tmp_111 = tmp_116 = None
        conv2d_15 = torch.conv2d(tmp_117, in_58, None, (1, 1), (0, 0), (1, 1), 1);  tmp_117 = in_58 = None
        tmp_119 = torch.nn.functional.batch_norm(conv2d_15, in_52, in_53, in_55, in_54, False, 0.1, 1e-05);  conv2d_15 = in_52 = in_53 = in_55 = in_54 = None
        conv2d_16 = torch.conv2d(tmp_119, in_73, None, (1, 1), (0, 0), (1, 1), 1);  tmp_119 = in_73 = None
        tmp_121 = torch.nn.functional.batch_norm(conv2d_16, in_67, in_68, in_70, in_69, False, 0.1, 1e-05);  conv2d_16 = in_67 = in_68 = in_70 = in_69 = None
        tmp_122 = torch.nn.functional.silu(tmp_121, inplace = True);  tmp_121 = None
        tmp_123 = torch.nn.functional.adaptive_avg_pool2d(tmp_122, 1);  tmp_122 = None
        tmp_124 = tmp_123.flatten(1, -1);  tmp_123 = None
        linear = torch.nn.functional.linear(tmp_124, in_72, in_71);  tmp_124 = in_72 = in_71 = None
        return (linear,)
        
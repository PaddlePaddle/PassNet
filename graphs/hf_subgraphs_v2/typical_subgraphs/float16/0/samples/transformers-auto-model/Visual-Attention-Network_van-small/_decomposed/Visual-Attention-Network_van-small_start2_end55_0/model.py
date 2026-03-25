import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor, in_29 : torch.Tensor, in_30 : torch.Tensor, in_31 : torch.Tensor, in_32 : torch.Tensor, in_33 : torch.Tensor, in_34 : torch.Tensor, in_35 : torch.Tensor, in_36 : torch.Tensor, in_37 : torch.Tensor, in_38 : torch.Tensor, in_39 : torch.Tensor, in_40 : torch.Tensor, in_41 : torch.Tensor, in_42 : torch.Tensor, in_43 : torch.Tensor, in_44 : torch.Tensor, in_45 : torch.Tensor, in_46 : torch.Tensor, in_47 : torch.Tensor, in_48 : torch.Tensor, in_49 : torch.Tensor, in_50 : torch.Tensor, in_51 : torch.Tensor, in_52 : torch.Tensor, in_53 : torch.Tensor, in_54 : torch.Tensor):
        tmp_54 = torch.nn.functional.batch_norm(in_54, in_22, in_23, in_25, in_24, False, 0.1, 1e-05);  in_22 = in_23 = in_25 = in_24 = None
        conv2d = torch.conv2d(tmp_54, in_9, in_8, (1, 1), (0, 0), (1, 1), 1);  in_9 = in_8 = None
        tmp_56 = torch.nn.functional.gelu(conv2d);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_56, in_3, in_2, (1, 1), (2, 2), (1, 1), 64);  in_3 = in_2 = None
        conv2d_2 = torch.conv2d(conv2d_1, in_1, in_0, (1, 1), (9, 9), (3, 3), 64);  conv2d_1 = in_1 = in_0 = None
        conv2d_3 = torch.conv2d(conv2d_2, in_5, in_4, (1, 1), (0, 0), (1, 1), 1);  conv2d_2 = in_5 = in_4 = None
        tmp_60 = tmp_56 * conv2d_3;  tmp_56 = conv2d_3 = None
        conv2d_4 = torch.conv2d(tmp_60, in_7, in_6, (1, 1), (0, 0), (1, 1), 1);  tmp_60 = in_7 = in_6 = None
        tmp_62 = conv2d_4 + tmp_54;  conv2d_4 = tmp_54 = None
        tmp_63 = in_10.unsqueeze(-1);  in_10 = None
        tmp_64 = tmp_63.unsqueeze(-1);  tmp_63 = None
        tmp_65 = tmp_64 * tmp_62;  tmp_64 = tmp_62 = None
        tmp_66 = in_54 + tmp_65;  in_54 = tmp_65 = None
        tmp_67 = torch.nn.functional.batch_norm(tmp_66, in_18, in_19, in_21, in_20, False, 0.1, 1e-05);  in_18 = in_19 = in_21 = in_20 = None
        conv2d_5 = torch.conv2d(tmp_67, in_14, in_13, (1, 1), (0, 0), (1, 1), 1);  tmp_67 = in_14 = in_13 = None
        conv2d_6 = torch.conv2d(conv2d_5, in_12, in_11, (1, 1), (1, 1), (1, 1), 512);  conv2d_5 = in_12 = in_11 = None
        tmp_70 = torch.nn.functional.gelu(conv2d_6);  conv2d_6 = None
        tmp_71 = torch.nn.functional.dropout(tmp_70, 0.0, False, False);  tmp_70 = None
        conv2d_7 = torch.conv2d(tmp_71, in_16, in_15, (1, 1), (0, 0), (1, 1), 1);  tmp_71 = in_16 = in_15 = None
        tmp_73 = torch.nn.functional.dropout(conv2d_7, 0.0, False, False);  conv2d_7 = None
        tmp_74 = in_17.unsqueeze(-1);  in_17 = None
        tmp_75 = tmp_74.unsqueeze(-1);  tmp_74 = None
        tmp_76 = tmp_75 * tmp_73;  tmp_75 = tmp_73 = None
        tmp_77 = tmp_66 + tmp_76;  tmp_66 = tmp_76 = None
        tmp_78 = torch.nn.functional.batch_norm(tmp_77, in_48, in_49, in_51, in_50, False, 0.1, 1e-05);  in_48 = in_49 = in_51 = in_50 = None
        conv2d_8 = torch.conv2d(tmp_78, in_35, in_34, (1, 1), (0, 0), (1, 1), 1);  in_35 = in_34 = None
        tmp_80 = torch.nn.functional.gelu(conv2d_8);  conv2d_8 = None
        conv2d_9 = torch.conv2d(tmp_80, in_29, in_28, (1, 1), (2, 2), (1, 1), 64);  in_29 = in_28 = None
        conv2d_10 = torch.conv2d(conv2d_9, in_27, in_26, (1, 1), (9, 9), (3, 3), 64);  conv2d_9 = in_27 = in_26 = None
        conv2d_11 = torch.conv2d(conv2d_10, in_31, in_30, (1, 1), (0, 0), (1, 1), 1);  conv2d_10 = in_31 = in_30 = None
        tmp_84 = tmp_80 * conv2d_11;  tmp_80 = conv2d_11 = None
        conv2d_12 = torch.conv2d(tmp_84, in_33, in_32, (1, 1), (0, 0), (1, 1), 1);  tmp_84 = in_33 = in_32 = None
        tmp_86 = conv2d_12 + tmp_78;  conv2d_12 = tmp_78 = None
        tmp_87 = in_36.unsqueeze(-1);  in_36 = None
        tmp_88 = tmp_87.unsqueeze(-1);  tmp_87 = None
        tmp_89 = tmp_88 * tmp_86;  tmp_88 = tmp_86 = None
        tmp_90 = tmp_77 + tmp_89;  tmp_77 = tmp_89 = None
        tmp_91 = torch.nn.functional.batch_norm(tmp_90, in_44, in_45, in_47, in_46, False, 0.1, 1e-05);  in_44 = in_45 = in_47 = in_46 = None
        conv2d_13 = torch.conv2d(tmp_91, in_40, in_39, (1, 1), (0, 0), (1, 1), 1);  tmp_91 = in_40 = in_39 = None
        conv2d_14 = torch.conv2d(conv2d_13, in_38, in_37, (1, 1), (1, 1), (1, 1), 512);  conv2d_13 = in_38 = in_37 = None
        tmp_94 = torch.nn.functional.gelu(conv2d_14);  conv2d_14 = None
        tmp_95 = torch.nn.functional.dropout(tmp_94, 0.0, False, False);  tmp_94 = None
        conv2d_15 = torch.conv2d(tmp_95, in_42, in_41, (1, 1), (0, 0), (1, 1), 1);  tmp_95 = in_42 = in_41 = None
        tmp_97 = torch.nn.functional.dropout(conv2d_15, 0.0, False, False);  conv2d_15 = None
        tmp_98 = in_43.unsqueeze(-1);  in_43 = None
        tmp_99 = tmp_98.unsqueeze(-1);  tmp_98 = None
        tmp_100 = tmp_99 * tmp_97;  tmp_99 = tmp_97 = None
        tmp_101 = tmp_90 + tmp_100;  tmp_90 = tmp_100 = None
        tmp_102 = tmp_101.flatten(2);  tmp_101 = None
        tmp_103 = tmp_102.transpose(1, 2);  tmp_102 = None
        tmp_104 = torch.nn.functional.layer_norm(tmp_103, (64,), in_53, in_52, 1e-06);  tmp_103 = in_53 = in_52 = None
        tmp_105 = tmp_104.view(1, 56, 56, 64);  tmp_104 = None
        tmp_106 = tmp_105.permute(0, 3, 1, 2);  tmp_105 = None
        return (tmp_106,)
        
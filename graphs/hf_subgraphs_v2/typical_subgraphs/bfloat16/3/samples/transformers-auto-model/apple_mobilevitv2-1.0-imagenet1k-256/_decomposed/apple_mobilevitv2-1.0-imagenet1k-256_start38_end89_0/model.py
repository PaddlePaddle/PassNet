import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor):
        tmp_27 = torch.nn.functional.silu(in_27, inplace = False);  in_27 = None
        conv2d = torch.conv2d(tmp_27, in_0, None, (1, 1), (0, 0), (1, 1), 1);  tmp_27 = in_0 = None
        tmp_29 = torch.nn.functional.unfold(conv2d, kernel_size = (2, 2), stride = (2, 2));  conv2d = None
        tmp_30 = tmp_29.reshape(1, 128, 4, -1);  tmp_29 = None
        tmp_31 = torch.nn.functional.group_norm(tmp_30, 1, in_14, in_13, 1e-05);  in_14 = in_13 = None
        conv2d_1 = torch.conv2d(tmp_31, in_6, in_5, (1, 1), (0, 0), (1, 1), 1);  tmp_31 = in_6 = in_5 = None
        split = torch.functional.split(conv2d_1, split_size_or_sections = [1, 128, 128], dim = 1);  conv2d_1 = None
        tmp_34 = split[0]
        tmp_35 = split[1]
        tmp_36 = split[2];  split = None
        tmp_37 = torch.nn.functional.softmax(tmp_34, dim = -1);  tmp_34 = None
        tmp_38 = torch.nn.functional.dropout(tmp_37, 0.0, False, False);  tmp_37 = None
        tmp_39 = tmp_35 * tmp_38;  tmp_35 = tmp_38 = None
        tmp_40 = torch.sum(tmp_39, dim = -1, keepdim = True);  tmp_39 = None
        tmp_41 = torch.nn.functional.relu(tmp_36)
        tmp_42 = tmp_40.expand_as(tmp_36);  tmp_40 = tmp_36 = None
        tmp_43 = tmp_41 * tmp_42;  tmp_41 = tmp_42 = None
        conv2d_2 = torch.conv2d(tmp_43, in_4, in_3, (1, 1), (0, 0), (1, 1), 1);  tmp_43 = in_4 = in_3 = None
        tmp_45 = conv2d_2 + tmp_30;  conv2d_2 = tmp_30 = None
        tmp_46 = torch.nn.functional.group_norm(tmp_45, 1, in_12, in_11, 1e-05);  in_12 = in_11 = None
        conv2d_3 = torch.conv2d(tmp_46, in_8, in_7, (1, 1), (0, 0), (1, 1), 1);  tmp_46 = in_8 = in_7 = None
        tmp_48 = torch.nn.functional.silu(conv2d_3, inplace = False);  conv2d_3 = None
        tmp_49 = torch.nn.functional.dropout(tmp_48, 0.0, False, False);  tmp_48 = None
        conv2d_4 = torch.conv2d(tmp_49, in_10, in_9, (1, 1), (0, 0), (1, 1), 1);  tmp_49 = in_10 = in_9 = None
        tmp_51 = torch.nn.functional.dropout(conv2d_4, 0.0, False, False);  conv2d_4 = None
        tmp_52 = tmp_51 + tmp_45;  tmp_51 = tmp_45 = None
        tmp_53 = torch.nn.functional.group_norm(tmp_52, 1, in_26, in_25, 1e-05);  in_26 = in_25 = None
        conv2d_5 = torch.conv2d(tmp_53, in_18, in_17, (1, 1), (0, 0), (1, 1), 1);  tmp_53 = in_18 = in_17 = None
        split_1 = torch.functional.split(conv2d_5, split_size_or_sections = [1, 128, 128], dim = 1);  conv2d_5 = None
        tmp_56 = split_1[0]
        tmp_57 = split_1[1]
        tmp_58 = split_1[2];  split_1 = None
        tmp_59 = torch.nn.functional.softmax(tmp_56, dim = -1);  tmp_56 = None
        tmp_60 = torch.nn.functional.dropout(tmp_59, 0.0, False, False);  tmp_59 = None
        tmp_61 = tmp_57 * tmp_60;  tmp_57 = tmp_60 = None
        tmp_62 = torch.sum(tmp_61, dim = -1, keepdim = True);  tmp_61 = None
        tmp_63 = torch.nn.functional.relu(tmp_58)
        tmp_64 = tmp_62.expand_as(tmp_58);  tmp_62 = tmp_58 = None
        tmp_65 = tmp_63 * tmp_64;  tmp_63 = tmp_64 = None
        conv2d_6 = torch.conv2d(tmp_65, in_16, in_15, (1, 1), (0, 0), (1, 1), 1);  tmp_65 = in_16 = in_15 = None
        tmp_67 = conv2d_6 + tmp_52;  conv2d_6 = tmp_52 = None
        tmp_68 = torch.nn.functional.group_norm(tmp_67, 1, in_24, in_23, 1e-05);  in_24 = in_23 = None
        conv2d_7 = torch.conv2d(tmp_68, in_20, in_19, (1, 1), (0, 0), (1, 1), 1);  tmp_68 = in_20 = in_19 = None
        tmp_70 = torch.nn.functional.silu(conv2d_7, inplace = False);  conv2d_7 = None
        tmp_71 = torch.nn.functional.dropout(tmp_70, 0.0, False, False);  tmp_70 = None
        conv2d_8 = torch.conv2d(tmp_71, in_22, in_21, (1, 1), (0, 0), (1, 1), 1);  tmp_71 = in_22 = in_21 = None
        tmp_73 = torch.nn.functional.dropout(conv2d_8, 0.0, False, False);  conv2d_8 = None
        tmp_74 = tmp_73 + tmp_67;  tmp_73 = tmp_67 = None
        tmp_75 = torch.nn.functional.group_norm(tmp_74, 1, in_2, in_1, 1e-05);  tmp_74 = in_2 = in_1 = None
        tmp_76 = tmp_75.reshape(1, 512, 256);  tmp_75 = None
        tmp_77 = torch.nn.functional.fold(tmp_76, output_size = (32, 32), kernel_size = (2, 2), stride = (2, 2));  tmp_76 = None
        return (tmp_77,)
        
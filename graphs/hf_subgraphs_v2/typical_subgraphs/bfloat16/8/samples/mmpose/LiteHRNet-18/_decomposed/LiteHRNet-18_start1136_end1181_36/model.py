import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23):
        tmp_16 = torch.nn.functional.adaptive_avg_pool2d(in_20, 1)
        conv2d = torch.conv2d(tmp_16, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_16 = in_1 = in_0 = None
        tmp_18 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_18, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  tmp_18 = in_3 = in_2 = None
        tmp_20 = torch.sigmoid(conv2d_1);  conv2d_1 = None
        tmp_21 = in_20 * tmp_20;  in_20 = tmp_20 = None
        tmp_22 = torch.nn.functional.adaptive_avg_pool2d(in_21, 1)
        conv2d_2 = torch.conv2d(tmp_22, in_5, in_4, (1, 1), (0, 0), (1, 1), 1);  tmp_22 = in_5 = in_4 = None
        tmp_24 = torch.nn.functional.relu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_24, in_7, in_6, (1, 1), (0, 0), (1, 1), 1);  tmp_24 = in_7 = in_6 = None
        tmp_26 = torch.sigmoid(conv2d_3);  conv2d_3 = None
        tmp_27 = in_21 * tmp_26;  in_21 = tmp_26 = None
        tmp_28 = torch.nn.functional.adaptive_avg_pool2d(in_22, 1)
        conv2d_4 = torch.conv2d(tmp_28, in_9, in_8, (1, 1), (0, 0), (1, 1), 1);  tmp_28 = in_9 = in_8 = None
        tmp_30 = torch.nn.functional.relu(conv2d_4, inplace = True);  conv2d_4 = None
        conv2d_5 = torch.conv2d(tmp_30, in_11, in_10, (1, 1), (0, 0), (1, 1), 1);  tmp_30 = in_11 = in_10 = None
        tmp_32 = torch.sigmoid(conv2d_5);  conv2d_5 = None
        tmp_33 = in_22 * tmp_32;  in_22 = tmp_32 = None
        tmp_34 = torch.nn.functional.adaptive_avg_pool2d(in_23, 1)
        conv2d_6 = torch.conv2d(tmp_34, in_13, in_12, (1, 1), (0, 0), (1, 1), 1);  tmp_34 = in_13 = in_12 = None
        tmp_36 = torch.nn.functional.relu(conv2d_6, inplace = True);  conv2d_6 = None
        conv2d_7 = torch.conv2d(tmp_36, in_15, in_14, (1, 1), (0, 0), (1, 1), 1);  tmp_36 = in_15 = in_14 = None
        tmp_38 = torch.sigmoid(conv2d_7);  conv2d_7 = None
        tmp_39 = in_23 * tmp_38;  in_23 = tmp_38 = None
        tmp_40 = torch.cat([in_16, tmp_21], dim = 1);  in_16 = tmp_21 = None
        tmp_41 = torch.cat([in_17, tmp_27], dim = 1);  in_17 = tmp_27 = None
        tmp_42 = torch.cat([in_18, tmp_33], dim = 1);  in_18 = tmp_33 = None
        tmp_43 = torch.cat([in_19, tmp_39], dim = 1);  in_19 = tmp_39 = None
        tmp_44 = tmp_40.view(512, 2, 20, 64, 48);  tmp_40 = None
        tmp_45 = torch.transpose(tmp_44, 1, 2);  tmp_44 = None
        tmp_46 = tmp_45.contiguous();  tmp_45 = None
        tmp_47 = tmp_46.view(512, 40, 64, 48);  tmp_46 = None
        tmp_48 = tmp_41.view(512, 2, 40, 32, 24);  tmp_41 = None
        tmp_49 = torch.transpose(tmp_48, 1, 2);  tmp_48 = None
        tmp_50 = tmp_49.contiguous();  tmp_49 = None
        tmp_51 = tmp_50.view(512, 80, 32, 24);  tmp_50 = None
        tmp_52 = tmp_42.view(512, 2, 80, 16, 12);  tmp_42 = None
        tmp_53 = torch.transpose(tmp_52, 1, 2);  tmp_52 = None
        tmp_54 = tmp_53.contiguous();  tmp_53 = None
        tmp_55 = tmp_54.view(512, 160, 16, 12);  tmp_54 = None
        tmp_56 = tmp_43.view(512, 2, 160, 8, 6);  tmp_43 = None
        tmp_57 = torch.transpose(tmp_56, 1, 2);  tmp_56 = None
        tmp_58 = tmp_57.contiguous();  tmp_57 = None
        tmp_59 = tmp_58.view(512, 320, 8, 6);  tmp_58 = None
        tmp_47 += tmp_47;  tmp_60 = tmp_47;  tmp_47 = None
        return (tmp_51, tmp_55, tmp_59, tmp_60)
        
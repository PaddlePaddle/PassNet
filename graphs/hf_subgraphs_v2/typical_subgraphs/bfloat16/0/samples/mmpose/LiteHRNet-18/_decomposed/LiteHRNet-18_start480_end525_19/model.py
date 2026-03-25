import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17):
        tmp_12 = torch.nn.functional.adaptive_avg_pool2d(in_15, 1)
        conv2d = torch.conv2d(tmp_12, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_12 = in_1 = in_0 = None
        tmp_14 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_14, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  tmp_14 = in_3 = in_2 = None
        tmp_16 = torch.sigmoid(conv2d_1);  conv2d_1 = None
        tmp_17 = in_15 * tmp_16;  in_15 = tmp_16 = None
        tmp_18 = torch.nn.functional.adaptive_avg_pool2d(in_16, 1)
        conv2d_2 = torch.conv2d(tmp_18, in_5, in_4, (1, 1), (0, 0), (1, 1), 1);  tmp_18 = in_5 = in_4 = None
        tmp_20 = torch.nn.functional.relu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_20, in_7, in_6, (1, 1), (0, 0), (1, 1), 1);  tmp_20 = in_7 = in_6 = None
        tmp_22 = torch.sigmoid(conv2d_3);  conv2d_3 = None
        tmp_23 = in_16 * tmp_22;  in_16 = tmp_22 = None
        tmp_24 = torch.nn.functional.adaptive_avg_pool2d(in_17, 1)
        conv2d_4 = torch.conv2d(tmp_24, in_9, in_8, (1, 1), (0, 0), (1, 1), 1);  tmp_24 = in_9 = in_8 = None
        tmp_26 = torch.nn.functional.relu(conv2d_4, inplace = True);  conv2d_4 = None
        conv2d_5 = torch.conv2d(tmp_26, in_11, in_10, (1, 1), (0, 0), (1, 1), 1);  tmp_26 = in_11 = in_10 = None
        tmp_28 = torch.sigmoid(conv2d_5);  conv2d_5 = None
        tmp_29 = in_17 * tmp_28;  in_17 = tmp_28 = None
        tmp_30 = torch.cat([in_12, tmp_17], dim = 1);  in_12 = tmp_17 = None
        tmp_31 = torch.cat([in_13, tmp_23], dim = 1);  in_13 = tmp_23 = None
        tmp_32 = torch.cat([in_14, tmp_29], dim = 1);  in_14 = tmp_29 = None
        tmp_33 = tmp_30.view(1, 2, 20, 64, 48);  tmp_30 = None
        tmp_34 = torch.transpose(tmp_33, 1, 2);  tmp_33 = None
        tmp_35 = tmp_34.contiguous();  tmp_34 = None
        tmp_36 = tmp_35.view(1, 40, 64, 48);  tmp_35 = None
        tmp_37 = tmp_31.view(1, 2, 40, 32, 24);  tmp_31 = None
        tmp_38 = torch.transpose(tmp_37, 1, 2);  tmp_37 = None
        tmp_39 = tmp_38.contiguous();  tmp_38 = None
        tmp_40 = tmp_39.view(1, 80, 32, 24);  tmp_39 = None
        tmp_41 = tmp_32.view(1, 2, 80, 16, 12);  tmp_32 = None
        tmp_42 = torch.transpose(tmp_41, 1, 2);  tmp_41 = None
        tmp_43 = tmp_42.contiguous();  tmp_42 = None
        tmp_44 = tmp_43.view(1, 160, 16, 12);  tmp_43 = None
        chunk = tmp_36.chunk(2, dim = 1);  tmp_36 = None
        tmp_46 = chunk[0]
        tmp_47 = chunk[1];  chunk = None
        chunk_1 = tmp_40.chunk(2, dim = 1);  tmp_40 = None
        tmp_49 = chunk_1[0]
        tmp_50 = chunk_1[1];  chunk_1 = None
        chunk_2 = tmp_44.chunk(2, dim = 1);  tmp_44 = None
        tmp_52 = chunk_2[0]
        tmp_53 = chunk_2[1];  chunk_2 = None
        tmp_54 = torch.nn.functional.adaptive_avg_pool2d(tmp_47, (16, 12))
        tmp_55 = torch.nn.functional.adaptive_avg_pool2d(tmp_50, (16, 12))
        tmp_56 = torch.cat([tmp_54, tmp_55, tmp_53], dim = 1);  tmp_54 = tmp_55 = None
        return (tmp_56, tmp_46, tmp_49, tmp_52, tmp_47, tmp_50, tmp_53)
        
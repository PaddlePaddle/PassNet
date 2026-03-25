import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_12 = torch.nn.functional.adaptive_avg_pool2d(in_3, 1)
        conv2d = torch.conv2d(tmp_12, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_12 = w_1 = w_0 = None
        tmp_14 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_14, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  tmp_14 = w_3 = w_2 = None
        tmp_16 = torch.sigmoid(conv2d_1);  conv2d_1 = None
        tmp_17 = in_3 * tmp_16;  in_3 = tmp_16 = None
        tmp_18 = torch.nn.functional.adaptive_avg_pool2d(in_4, 1)
        conv2d_2 = torch.conv2d(tmp_18, w_5, w_4, (1, 1), (0, 0), (1, 1), 1);  tmp_18 = w_5 = w_4 = None
        tmp_20 = torch.nn.functional.relu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_20, w_7, w_6, (1, 1), (0, 0), (1, 1), 1);  tmp_20 = w_7 = w_6 = None
        tmp_22 = torch.sigmoid(conv2d_3);  conv2d_3 = None
        tmp_23 = in_4 * tmp_22;  in_4 = tmp_22 = None
        tmp_24 = torch.nn.functional.adaptive_avg_pool2d(in_5, 1)
        conv2d_4 = torch.conv2d(tmp_24, w_9, w_8, (1, 1), (0, 0), (1, 1), 1);  tmp_24 = w_9 = w_8 = None
        tmp_26 = torch.nn.functional.relu(conv2d_4, inplace = True);  conv2d_4 = None
        conv2d_5 = torch.conv2d(tmp_26, w_11, w_10, (1, 1), (0, 0), (1, 1), 1);  tmp_26 = w_11 = w_10 = None
        tmp_28 = torch.sigmoid(conv2d_5);  conv2d_5 = None
        tmp_29 = in_5 * tmp_28;  in_5 = tmp_28 = None
        tmp_30 = torch.cat([in_0, tmp_17], dim = 1);  in_0 = tmp_17 = None
        tmp_31 = torch.cat([in_1, tmp_23], dim = 1);  in_1 = tmp_23 = None
        tmp_32 = torch.cat([in_2, tmp_29], dim = 1);  in_2 = tmp_29 = None
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
        tmp_36 += tmp_36;  tmp_45 = tmp_36;  tmp_36 = None
        return (tmp_40, tmp_44, tmp_45)
        
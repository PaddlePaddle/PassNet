import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0, in_1, in_2, in_3):
        tmp_8 = torch.nn.functional.adaptive_avg_pool2d(in_3, 1)
        conv2d = torch.conv2d(tmp_8, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_8 = w_1 = w_0 = None
        tmp_10 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_10, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  tmp_10 = w_3 = w_2 = None
        tmp_12 = torch.sigmoid(conv2d_1);  conv2d_1 = None
        tmp_13 = in_3 * tmp_12;  in_3 = tmp_12 = None
        tmp_14 = torch.nn.functional.adaptive_avg_pool2d(in_2, 1)
        conv2d_2 = torch.conv2d(tmp_14, w_5, w_4, (1, 1), (0, 0), (1, 1), 1);  tmp_14 = w_5 = w_4 = None
        tmp_16 = torch.nn.functional.relu(conv2d_2, inplace = True);  conv2d_2 = None
        conv2d_3 = torch.conv2d(tmp_16, w_7, w_6, (1, 1), (0, 0), (1, 1), 1);  tmp_16 = w_7 = w_6 = None
        tmp_18 = torch.sigmoid(conv2d_3);  conv2d_3 = None
        tmp_19 = in_2 * tmp_18;  in_2 = tmp_18 = None
        tmp_20 = torch.cat([in_0, tmp_13], dim = 1);  in_0 = tmp_13 = None
        tmp_21 = torch.cat([in_1, tmp_19], dim = 1);  in_1 = tmp_19 = None
        tmp_22 = tmp_20.view(1, 2, 20, 64, 48);  tmp_20 = None
        tmp_23 = torch.transpose(tmp_22, 1, 2);  tmp_22 = None
        tmp_24 = tmp_23.contiguous();  tmp_23 = None
        tmp_25 = tmp_24.view(1, 40, 64, 48);  tmp_24 = None
        tmp_26 = tmp_21.view(1, 2, 40, 32, 24);  tmp_21 = None
        tmp_27 = torch.transpose(tmp_26, 1, 2);  tmp_26 = None
        tmp_28 = tmp_27.contiguous();  tmp_27 = None
        tmp_29 = tmp_28.view(1, 80, 32, 24);  tmp_28 = None
        tmp_25 += tmp_25;  tmp_30 = tmp_25;  tmp_25 = None
        return (tmp_29, tmp_30)
        
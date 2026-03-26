import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        conv2d = torch.conv2d(in_8, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_8 = in_1 = in_0 = None
        tmp_3 = torch.sigmoid(conv2d);  conv2d = None
        tmp_4 = in_7 * tmp_3;  in_7 = tmp_3 = None
        tmp_5 = torch.cat([in_2, in_5], dim = 1);  in_2 = in_5 = None
        tmp_6 = torch.cat([in_3, in_6], dim = 1);  in_3 = in_6 = None
        tmp_7 = torch.cat([in_4, tmp_4], dim = 1);  in_4 = tmp_4 = None
        tmp_8 = tmp_5.view(256, 2, 20, 64, 48);  tmp_5 = None
        tmp_9 = torch.transpose(tmp_8, 1, 2);  tmp_8 = None
        tmp_10 = tmp_9.contiguous();  tmp_9 = None
        tmp_11 = tmp_10.view(256, 40, 64, 48);  tmp_10 = None
        tmp_12 = tmp_6.view(256, 2, 40, 32, 24);  tmp_6 = None
        tmp_13 = torch.transpose(tmp_12, 1, 2);  tmp_12 = None
        tmp_14 = tmp_13.contiguous();  tmp_13 = None
        tmp_15 = tmp_14.view(256, 80, 32, 24);  tmp_14 = None
        tmp_16 = tmp_7.view(256, 2, 80, 16, 12);  tmp_7 = None
        tmp_17 = torch.transpose(tmp_16, 1, 2);  tmp_16 = None
        tmp_18 = tmp_17.contiguous();  tmp_17 = None
        tmp_19 = tmp_18.view(256, 160, 16, 12);  tmp_18 = None
        tmp_11 += tmp_11;  tmp_20 = tmp_11;  tmp_11 = None
        return (tmp_15, tmp_19, tmp_20)
        
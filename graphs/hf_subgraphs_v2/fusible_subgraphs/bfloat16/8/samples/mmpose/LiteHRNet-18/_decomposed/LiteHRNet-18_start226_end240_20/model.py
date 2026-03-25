import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        conv2d = torch.conv2d(in_6, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_6 = in_1 = in_0 = None
        tmp_3 = torch.sigmoid(conv2d);  conv2d = None
        tmp_4 = in_5 * tmp_3;  in_5 = tmp_3 = None
        tmp_5 = torch.cat([in_2, in_4], dim = 1);  in_2 = in_4 = None
        tmp_6 = torch.cat([in_3, tmp_4], dim = 1);  in_3 = tmp_4 = None
        tmp_7 = tmp_5.view(512, 2, 20, 64, 48);  tmp_5 = None
        tmp_8 = torch.transpose(tmp_7, 1, 2);  tmp_7 = None
        tmp_9 = tmp_8.contiguous();  tmp_8 = None
        tmp_10 = tmp_9.view(512, 40, 64, 48);  tmp_9 = None
        tmp_11 = tmp_6.view(512, 2, 40, 32, 24);  tmp_6 = None
        tmp_12 = torch.transpose(tmp_11, 1, 2);  tmp_11 = None
        tmp_13 = tmp_12.contiguous();  tmp_12 = None
        tmp_14 = tmp_13.view(512, 80, 32, 24);  tmp_13 = None
        tmp_10 += tmp_10;  tmp_15 = tmp_10;  tmp_10 = None
        return (tmp_14, tmp_15)
        
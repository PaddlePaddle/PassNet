import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor):
        tmp_10 = torch.nn.functional.silu(in_10, inplace = True);  in_10 = None
        conv2d = torch.conv2d(in_11, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_11 = in_1 = in_0 = None
        split = conv2d.split([64], dim = 1);  conv2d = None
        tmp_13 = split[0];  split = None
        conv2d_1 = torch.conv2d(in_12, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  in_12 = in_3 = in_2 = None
        split_1 = conv2d_1.split([64, 128], dim = 1);  conv2d_1 = None
        tmp_16 = split_1[0]
        tmp_17 = split_1[1];  split_1 = None
        conv2d_2 = torch.conv2d(in_13, in_5, in_4, (1, 1), (0, 0), (1, 1), 1);  in_13 = in_5 = in_4 = None
        split_2 = conv2d_2.split([64, 128, 256], dim = 1);  conv2d_2 = None
        tmp_20 = split_2[0]
        tmp_21 = split_2[1]
        tmp_22 = split_2[2];  split_2 = None
        conv2d_3 = torch.conv2d(in_14, in_7, in_6, (1, 1), (0, 0), (1, 1), 1);  in_14 = in_7 = in_6 = None
        split_3 = conv2d_3.split([64, 128, 256, 512], dim = 1);  conv2d_3 = None
        tmp_25 = split_3[0]
        tmp_26 = split_3[1]
        tmp_27 = split_3[2]
        tmp_28 = split_3[3];  split_3 = None
        conv2d_4 = torch.conv2d(tmp_10, in_9, in_8, (1, 1), (0, 0), (1, 1), 1);  tmp_10 = in_9 = in_8 = None
        split_4 = conv2d_4.split([64, 128, 256, 512, 1024], dim = 1);  conv2d_4 = None
        tmp_31 = split_4[0]
        tmp_32 = split_4[1]
        tmp_33 = split_4[2]
        tmp_34 = split_4[3]
        tmp_35 = split_4[4];  split_4 = None
        return (tmp_13, tmp_16, tmp_17, tmp_20, tmp_21, tmp_22, tmp_25, tmp_26, tmp_27, tmp_28, tmp_31, tmp_32, tmp_33, tmp_34, tmp_35)
        
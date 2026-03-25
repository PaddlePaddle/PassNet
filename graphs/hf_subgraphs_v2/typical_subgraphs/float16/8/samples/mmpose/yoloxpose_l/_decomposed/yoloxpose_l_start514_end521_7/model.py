import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12):
        tmp_10 = torch.sigmoid(in_12)
        tmp_11 = in_12 * tmp_10;  in_12 = tmp_10 = None
        conv2d = torch.conv2d(in_10, in_3, in_2, (1, 1), (0, 0), (1, 1), 1);  in_10 = in_3 = in_2 = None
        conv2d_1 = torch.conv2d(in_11, in_9, in_8, (1, 1), (0, 0), (1, 1), 1);  in_9 = in_8 = None
        conv2d_2 = torch.conv2d(in_11, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_11 = in_1 = in_0 = None
        conv2d_3 = torch.conv2d(tmp_11, in_5, in_4, (1, 1), (0, 0), (1, 1), 1);  in_5 = in_4 = None
        conv2d_4 = torch.conv2d(tmp_11, in_7, in_6, (1, 1), (0, 0), (1, 1), 1);  tmp_11 = in_7 = in_6 = None
        return (conv2d, conv2d_1, conv2d_2, conv2d_3, conv2d_4)
        
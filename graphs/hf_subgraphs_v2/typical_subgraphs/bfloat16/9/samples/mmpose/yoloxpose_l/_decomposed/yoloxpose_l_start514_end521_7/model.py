import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1, in_2):
        tmp_10 = torch.sigmoid(in_2)
        tmp_11 = in_2 * tmp_10;  in_2 = tmp_10 = None
        conv2d = torch.conv2d(in_0, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  in_0 = w_3 = w_2 = None
        conv2d_1 = torch.conv2d(in_1, w_9, w_8, (1, 1), (0, 0), (1, 1), 1);  w_9 = w_8 = None
        conv2d_2 = torch.conv2d(in_1, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  in_1 = w_1 = w_0 = None
        conv2d_3 = torch.conv2d(tmp_11, w_5, w_4, (1, 1), (0, 0), (1, 1), 1);  w_5 = w_4 = None
        conv2d_4 = torch.conv2d(tmp_11, w_7, w_6, (1, 1), (0, 0), (1, 1), 1);  tmp_11 = w_7 = w_6 = None
        return (conv2d, conv2d_1, conv2d_2, conv2d_3, conv2d_4)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0, in_1):
        tmp_8 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        conv2d = torch.conv2d(tmp_8, w_1, None, (2, 2), (1, 1), (1, 1), 256);  tmp_8 = w_1 = None
        conv2d_1 = torch.conv2d(conv2d, w_2, None, (1, 1), (0, 0), (1, 1), 1);  conv2d = w_2 = None
        conv2d_2 = torch.conv2d(in_0, w_0, None, (2, 2), (0, 0), (1, 1), 1);  in_0 = w_0 = None
        tmp_12 = conv2d_1 + conv2d_2;  conv2d_1 = conv2d_2 = None
        tmp_13 = torch.nn.functional.batch_norm(tmp_12, w_3, w_4, w_6, w_5, False, 0.1, 0.001);  tmp_12 = w_3 = w_4 = w_6 = w_5 = None
        tmp_14 = torch.nn.functional.relu(tmp_13, inplace = True);  tmp_13 = None
        conv2d_3 = torch.conv2d(tmp_14, w_7, None, (1, 1), (1, 1), (1, 1), 256);  w_7 = None
        return (tmp_14, conv2d_3)
        
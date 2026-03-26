import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1, in_2):
        tmp_6 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        conv2d = torch.conv2d(tmp_6, w_0, None, (1, 1), (0, 0), (1, 1), 1);  w_0 = None
        conv2d_1 = torch.conv2d(tmp_6, w_1, None, (1, 1), (0, 0), (1, 1), 1);  tmp_6 = w_1 = None
        tmp_9 = in_1 + conv2d;  in_1 = conv2d = None
        tmp_10 = torch.cat([in_0, conv2d_1], dim = 1);  in_0 = conv2d_1 = None
        tmp_11 = torch.cat((tmp_9, tmp_10), dim = 1)
        tmp_12 = torch.nn.functional.batch_norm(tmp_11, w_2, w_3, w_5, w_4, False, 0.1, 0.001);  tmp_11 = w_2 = w_3 = w_5 = w_4 = None
        return (tmp_10, tmp_9, tmp_12)
        
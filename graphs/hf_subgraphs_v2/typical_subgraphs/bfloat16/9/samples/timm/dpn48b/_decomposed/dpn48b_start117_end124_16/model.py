import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_6 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        conv2d = torch.conv2d(tmp_6, w_0, None, (1, 1), (0, 0), (1, 1), 1);  w_0 = None
        conv2d_1 = torch.conv2d(tmp_6, w_1, None, (1, 1), (0, 0), (1, 1), 1);  tmp_6 = w_1 = None
        tmp_9 = in_1 + conv2d;  in_1 = conv2d = None
        tmp_10 = torch.cat([in_2, conv2d_1], dim = 1);  in_2 = conv2d_1 = None
        tmp_11 = torch.cat((tmp_9, tmp_10), dim = 1)
        tmp_12 = torch.nn.functional.batch_norm(tmp_11, w_2, w_3, w_5, w_4, False, 0.1, 0.001);  tmp_11 = w_2 = w_3 = w_5 = w_4 = None
        return (tmp_10, tmp_9, tmp_12)
        
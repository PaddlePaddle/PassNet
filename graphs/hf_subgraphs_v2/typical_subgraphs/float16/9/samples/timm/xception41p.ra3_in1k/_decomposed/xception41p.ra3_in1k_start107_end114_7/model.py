import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_7 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        conv2d = torch.conv2d(tmp_7, w_0, None, (1, 1), (1, 1), (1, 1), 728);  tmp_7 = w_0 = None
        conv2d_1 = torch.conv2d(conv2d, w_1, None, (1, 1), (0, 0), (1, 1), 1);  conv2d = w_1 = None
        tmp_10 = conv2d_1 + in_1;  conv2d_1 = in_1 = None
        tmp_11 = torch.nn.functional.batch_norm(tmp_10, w_2, w_3, w_5, w_4, False, 0.1, 1e-05);  tmp_10 = w_2 = w_3 = w_5 = w_4 = None
        tmp_12 = torch.nn.functional.relu(tmp_11, inplace = True);  tmp_11 = None
        conv2d_2 = torch.conv2d(tmp_12, w_6, None, (1, 1), (1, 1), (1, 1), 728);  w_6 = None
        return (tmp_12, conv2d_2)
        
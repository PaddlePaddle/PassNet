import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, in_0, in_1):
        tmp_7 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        conv2d = torch.conv2d(tmp_7, w_5, None, (1, 1), (1, 1), (1, 1), 728);  tmp_7 = w_5 = None
        conv2d_1 = torch.conv2d(conv2d, w_6, None, (1, 1), (0, 0), (1, 1), 1);  conv2d = w_6 = None
        tmp_10 = conv2d_1 + in_0;  conv2d_1 = in_0 = None
        tmp_11 = torch.nn.functional.batch_norm(tmp_10, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  tmp_10 = w_0 = w_1 = w_3 = w_2 = None
        tmp_12 = torch.nn.functional.relu(tmp_11, inplace = True);  tmp_11 = None
        conv2d_2 = torch.conv2d(tmp_12, w_4, None, (1, 1), (1, 1), (1, 1), 728);  w_4 = None
        return (tmp_12, conv2d_2)
        
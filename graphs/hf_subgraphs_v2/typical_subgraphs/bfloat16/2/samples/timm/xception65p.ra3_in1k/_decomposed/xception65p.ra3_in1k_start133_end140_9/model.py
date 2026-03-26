import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        tmp_7 = torch.nn.functional.relu(in_8, inplace = True);  in_8 = None
        conv2d = torch.conv2d(tmp_7, in_5, None, (1, 1), (1, 1), (1, 1), 728);  tmp_7 = in_5 = None
        conv2d_1 = torch.conv2d(conv2d, in_6, None, (1, 1), (0, 0), (1, 1), 1);  conv2d = in_6 = None
        tmp_10 = conv2d_1 + in_7;  conv2d_1 = in_7 = None
        tmp_11 = torch.nn.functional.batch_norm(tmp_10, in_0, in_1, in_3, in_2, False, 0.1, 0.001);  tmp_10 = in_0 = in_1 = in_3 = in_2 = None
        tmp_12 = torch.nn.functional.relu(tmp_11, inplace = True);  tmp_11 = None
        conv2d_2 = torch.conv2d(tmp_12, in_4, None, (1, 1), (1, 1), (1, 1), 728);  in_4 = None
        return (tmp_12, conv2d_2)
        
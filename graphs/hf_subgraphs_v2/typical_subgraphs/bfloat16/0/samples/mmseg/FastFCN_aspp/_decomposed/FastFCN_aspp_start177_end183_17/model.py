import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12):
        tmp_10 = torch.nn.functional.relu(in_11, inplace = True);  in_11 = None
        conv2d = torch.conv2d(in_12, in_4, None, (1, 1), (1, 1), (1, 1), 1);  in_12 = in_4 = None
        tmp_12 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  conv2d = in_0 = in_1 = in_3 = in_2 = None
        tmp_13 = torch.nn.functional.relu(tmp_12, inplace = True);  tmp_12 = None
        conv2d_1 = torch.conv2d(in_10, in_9, None, (1, 1), (1, 1), (1, 1), 1);  in_10 = in_9 = None
        tmp_15 = torch.nn.functional.batch_norm(conv2d_1, in_5, in_6, in_8, in_7, False, 0.1, 1e-05);  conv2d_1 = in_5 = in_6 = in_8 = in_7 = None
        return (tmp_10, tmp_13, tmp_15)
        
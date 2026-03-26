import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1):
        tmp_10 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        conv2d = torch.conv2d(in_0, w_4, None, (1, 1), (12, 12), (12, 12), 512);  in_0 = w_4 = None
        tmp_12 = torch.nn.functional.batch_norm(conv2d, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  conv2d = w_0 = w_1 = w_3 = w_2 = None
        tmp_13 = torch.nn.functional.relu(tmp_12, inplace = True);  tmp_12 = None
        conv2d_1 = torch.conv2d(tmp_13, w_9, None, (1, 1), (0, 0), (1, 1), 1);  tmp_13 = w_9 = None
        tmp_15 = torch.nn.functional.batch_norm(conv2d_1, w_5, w_6, w_8, w_7, False, 0.1, 1e-05);  conv2d_1 = w_5 = w_6 = w_8 = w_7 = None
        return (tmp_15, tmp_10)
        
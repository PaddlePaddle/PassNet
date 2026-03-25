import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
        tmp_10 = torch.nn.functional.relu(in_11, inplace = False);  in_11 = None
        conv2d = torch.conv2d(in_10, in_0, None, (1, 1), (12, 12), (12, 12), 1);  in_0 = None
        tmp_12 = torch.nn.functional.batch_norm(conv2d, in_1, in_2, in_4, in_3, False, 0.1, 1e-05);  conv2d = in_1 = in_2 = in_4 = in_3 = None
        tmp_13 = torch.nn.functional.relu(tmp_12, inplace = False);  tmp_12 = None
        conv2d_1 = torch.conv2d(in_10, in_5, None, (1, 1), (24, 24), (24, 24), 1);  in_10 = in_5 = None
        tmp_15 = torch.nn.functional.batch_norm(conv2d_1, in_6, in_7, in_9, in_8, False, 0.1, 1e-05);  conv2d_1 = in_6 = in_7 = in_9 = in_8 = None
        return (tmp_10, tmp_13, tmp_15)
        
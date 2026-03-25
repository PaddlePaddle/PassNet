import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        tmp_8 = torch.nn.functional.relu(in_9, inplace = True);  in_9 = None
        conv2d = torch.conv2d(tmp_8, in_1, None, (2, 2), (1, 1), (1, 1), 1024);  tmp_8 = in_1 = None
        conv2d_1 = torch.conv2d(conv2d, in_2, None, (1, 1), (0, 0), (1, 1), 1);  conv2d = in_2 = None
        conv2d_2 = torch.conv2d(in_8, in_0, None, (2, 2), (0, 0), (1, 1), 1);  in_8 = in_0 = None
        tmp_12 = conv2d_1 + conv2d_2;  conv2d_1 = conv2d_2 = None
        tmp_13 = torch.nn.functional.batch_norm(tmp_12, in_3, in_4, in_6, in_5, False, 0.1, 1e-05);  tmp_12 = in_3 = in_4 = in_6 = in_5 = None
        tmp_14 = torch.nn.functional.relu(tmp_13, inplace = True);  tmp_13 = None
        conv2d_3 = torch.conv2d(tmp_14, in_7, None, (1, 1), (1, 1), (1, 1), 1024);  tmp_14 = in_7 = None
        return (conv2d_3,)
        
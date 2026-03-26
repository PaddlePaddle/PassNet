import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_6 = torch.nn.functional.relu(in_7, inplace = True);  in_7 = None
        conv2d = torch.conv2d(tmp_6, in_0, None, (1, 1), (1, 1), (1, 1), 1);  tmp_6 = in_0 = None
        tmp_8 = conv2d + in_6;  conv2d = in_6 = None
        tmp_9 = torch.nn.functional.batch_norm(tmp_8, in_2, in_3, in_5, in_4, False, 0.1, 1e-05);  tmp_8 = in_2 = in_3 = in_5 = in_4 = None
        tmp_10 = torch.nn.functional.relu(tmp_9, inplace = True);  tmp_9 = None
        tmp_11 = torch.nn.functional.avg_pool2d(tmp_10, 2, 2, 0, True, False, None)
        conv2d_1 = torch.conv2d(tmp_11, in_1, None, (1, 1), (0, 0), (1, 1), 1);  tmp_11 = in_1 = None
        return (conv2d_1, tmp_10)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        tmp_7 = torch.nn.functional.relu(in_8, inplace = True);  in_8 = None
        conv2d = torch.conv2d(tmp_7, in_6, None, (1, 1), (0, 0), (1, 1), 1);  tmp_7 = in_6 = None
        tmp_9 = conv2d + in_7;  conv2d = in_7 = None
        tmp_10 = torch.nn.functional.batch_norm(tmp_9, in_2, in_3, in_5, in_4, False, 0.1, 1e-05);  tmp_9 = in_2 = in_3 = in_5 = in_4 = None
        tmp_11 = torch.nn.functional.relu(tmp_10, inplace = True);  tmp_10 = None
        tmp_12 = torch.nn.functional.adaptive_avg_pool2d(tmp_11, 1);  tmp_11 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False);  tmp_12 = None
        conv2d_1 = torch.conv2d(tmp_13, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_13 = in_1 = in_0 = None
        tmp_15 = conv2d_1.flatten(1, -1);  conv2d_1 = None
        return (tmp_15,)
        
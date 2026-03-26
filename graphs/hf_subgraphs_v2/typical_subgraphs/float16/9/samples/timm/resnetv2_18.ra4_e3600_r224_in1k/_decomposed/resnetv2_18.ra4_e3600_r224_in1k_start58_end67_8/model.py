import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, in_0, in_1):
        tmp_7 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        conv2d = torch.conv2d(tmp_7, w_6, None, (1, 1), (1, 1), (1, 1), 1);  tmp_7 = w_6 = None
        tmp_9 = conv2d + in_0;  conv2d = in_0 = None
        tmp_10 = torch.nn.functional.batch_norm(tmp_9, w_2, w_3, w_5, w_4, False, 0.1, 1e-05);  tmp_9 = w_2 = w_3 = w_5 = w_4 = None
        tmp_11 = torch.nn.functional.relu(tmp_10, inplace = True);  tmp_10 = None
        tmp_12 = torch.nn.functional.adaptive_avg_pool2d(tmp_11, 1);  tmp_11 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False);  tmp_12 = None
        conv2d_1 = torch.conv2d(tmp_13, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_13 = w_1 = w_0 = None
        tmp_15 = conv2d_1.flatten(1, -1);  conv2d_1 = None
        return (tmp_15,)
        
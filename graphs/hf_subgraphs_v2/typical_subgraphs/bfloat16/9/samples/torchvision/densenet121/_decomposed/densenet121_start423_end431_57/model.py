import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16):
        tmp_7 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        conv2d = torch.conv2d(tmp_7, w_2, None, (1, 1), (1, 1), (1, 1), 1);  tmp_7 = w_2 = None
        tmp_9 = torch.cat([in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, conv2d], 1);  in_1 = in_2 = in_3 = in_4 = in_5 = in_6 = in_7 = in_8 = in_9 = in_10 = in_11 = in_12 = in_13 = in_14 = in_15 = in_16 = conv2d = None
        tmp_10 = torch.nn.functional.batch_norm(tmp_9, w_3, w_4, w_6, w_5, False, 0.1, 1e-05);  tmp_9 = w_3 = w_4 = w_6 = w_5 = None
        tmp_11 = torch.nn.functional.relu(tmp_10, inplace = True);  tmp_10 = None
        tmp_12 = torch.nn.functional.adaptive_avg_pool2d(tmp_11, (1, 1));  tmp_11 = None
        tmp_13 = torch.flatten(tmp_12, 1);  tmp_12 = None
        linear = torch.nn.functional.linear(tmp_13, w_1, w_0);  tmp_13 = w_1 = w_0 = None
        return (linear,)
        
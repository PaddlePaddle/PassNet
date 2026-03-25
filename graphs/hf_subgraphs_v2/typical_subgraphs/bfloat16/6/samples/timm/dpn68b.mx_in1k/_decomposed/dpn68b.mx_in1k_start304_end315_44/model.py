import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
        tmp_8 = torch.nn.functional.relu(in_10, inplace = True);  in_10 = None
        conv2d = torch.conv2d(tmp_8, in_2, None, (1, 1), (0, 0), (1, 1), 1);  in_2 = None
        conv2d_1 = torch.conv2d(tmp_8, in_3, None, (1, 1), (0, 0), (1, 1), 1);  tmp_8 = in_3 = None
        tmp_11 = in_9 + conv2d;  in_9 = conv2d = None
        tmp_12 = torch.cat([in_8, conv2d_1], dim = 1);  in_8 = conv2d_1 = None
        tmp_13 = torch.cat((tmp_11, tmp_12), dim = 1);  tmp_11 = tmp_12 = None
        tmp_14 = torch.nn.functional.batch_norm(tmp_13, in_4, in_5, in_7, in_6, False, 0.1, 0.001);  tmp_13 = in_4 = in_5 = in_7 = in_6 = None
        tmp_15 = torch.nn.functional.relu(tmp_14, inplace = False);  tmp_14 = None
        tmp_16 = torch.nn.functional.adaptive_avg_pool2d(tmp_15, 1);  tmp_15 = None
        conv2d_2 = torch.conv2d(tmp_16, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_16 = in_1 = in_0 = None
        tmp_18 = conv2d_2.flatten(1, -1);  conv2d_2 = None
        return (tmp_18,)
        
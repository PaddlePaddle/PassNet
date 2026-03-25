import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, in_0, in_1, in_2):
        tmp_7 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        conv2d = torch.conv2d(tmp_7, w_2, None, (1, 1), (0, 0), (1, 1), 1);  tmp_7 = w_2 = None
        tmp_9 = conv2d[(slice(None, None, None), slice(None, 512, None), slice(None, None, None), slice(None, None, None))]
        tmp_10 = conv2d[(slice(None, None, None), slice(512, None, None), slice(None, None, None), slice(None, None, None))];  conv2d = None
        tmp_11 = in_1 + tmp_9;  in_1 = tmp_9 = None
        tmp_12 = torch.cat([in_0, tmp_10], dim = 1);  in_0 = tmp_10 = None
        tmp_13 = torch.cat((tmp_11, tmp_12), dim = 1);  tmp_11 = tmp_12 = None
        tmp_14 = torch.nn.functional.batch_norm(tmp_13, w_3, w_4, w_6, w_5, False, 0.1, 0.001);  tmp_13 = w_3 = w_4 = w_6 = w_5 = None
        tmp_15 = torch.nn.functional.relu(tmp_14, inplace = False);  tmp_14 = None
        tmp_16 = torch.nn.functional.adaptive_avg_pool2d(tmp_15, 1);  tmp_15 = None
        conv2d_1 = torch.conv2d(tmp_16, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_16 = w_1 = w_0 = None
        tmp_18 = conv2d_1.flatten(1, -1);  conv2d_1 = None
        return (tmp_18,)
        
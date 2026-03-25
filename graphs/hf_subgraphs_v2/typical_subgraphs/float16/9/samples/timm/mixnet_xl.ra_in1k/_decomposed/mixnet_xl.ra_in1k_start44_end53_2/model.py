import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_6 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        split = torch.functional.split(tmp_6, [72, 72], 1);  tmp_6 = None
        tmp_8 = split[0]
        tmp_9 = split[1];  split = None
        conv2d = torch.conv2d(tmp_8, w_4, None, (1, 1), (0, 0), (1, 1), 1);  tmp_8 = w_4 = None
        conv2d_1 = torch.conv2d(tmp_9, w_5, None, (1, 1), (0, 0), (1, 1), 1);  tmp_9 = w_5 = None
        tmp_12 = torch.cat([conv2d, conv2d_1], 1);  conv2d = conv2d_1 = None
        tmp_13 = torch.nn.functional.batch_norm(tmp_12, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  tmp_12 = w_0 = w_1 = w_3 = w_2 = None
        tmp_14 = tmp_13 + in_0;  tmp_13 = in_0 = None
        return (tmp_14,)
        
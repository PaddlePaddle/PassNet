import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_6 = in_7 + in_6;  in_7 = in_6 = None
        split = torch.functional.split(tmp_6, [8, 8], 1);  tmp_6 = None
        tmp_8 = split[0]
        tmp_9 = split[1];  split = None
        conv2d = torch.conv2d(tmp_8, in_4, None, (1, 1), (0, 0), (1, 1), 1);  tmp_8 = in_4 = None
        conv2d_1 = torch.conv2d(tmp_9, in_5, None, (1, 1), (0, 0), (1, 1), 1);  tmp_9 = in_5 = None
        tmp_12 = torch.cat([conv2d, conv2d_1], 1);  conv2d = conv2d_1 = None
        tmp_13 = torch.nn.functional.batch_norm(tmp_12, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_12 = in_0 = in_1 = in_3 = in_2 = None
        tmp_14 = torch.nn.functional.relu(tmp_13, inplace = True);  tmp_13 = None
        return (tmp_14,)
        
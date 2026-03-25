import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (0, 0), (1, 1), 1);  in_6 = in_4 = None
        tmp_6 = torch.cat([in_5, conv2d], 1);  in_5 = conv2d = None
        tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_6 = in_0 = in_1 = in_3 = in_2 = None
        tmp_8 = torch.nn.functional.relu(tmp_7, inplace = True);  tmp_7 = None
        split = torch.functional.split(tmp_8, [80, 80, 80], 1);  tmp_8 = None
        tmp_10 = split[0]
        tmp_11 = split[1]
        tmp_12 = split[2];  split = None
        return (tmp_10, tmp_11, tmp_12)
        
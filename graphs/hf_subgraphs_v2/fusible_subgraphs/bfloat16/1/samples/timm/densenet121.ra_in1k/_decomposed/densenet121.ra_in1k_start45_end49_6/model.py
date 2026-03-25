import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
        conv2d = torch.conv2d(in_11, in_0, None, (1, 1), (1, 1), (1, 1), 1);  in_11 = in_0 = None
        tmp_6 = torch.cat([in_5, in_6, in_7, in_8, in_9, in_10, conv2d], 1);  in_5 = in_6 = in_7 = in_8 = in_9 = in_10 = conv2d = None
        tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_1, in_2, in_4, in_3, False, 0.1, 1e-05);  tmp_6 = in_1 = in_2 = in_4 = in_3 = None
        tmp_8 = torch.nn.functional.relu(tmp_7, inplace = True);  tmp_7 = None
        return (tmp_8,)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        tmp_5 = torch.nn.functional.relu(in_5, inplace = True);  in_5 = None
        conv2d = torch.conv2d(tmp_5, in_0, None, (1, 1), (1, 1), (1, 1), 1);  tmp_5 = in_0 = None
        tmp_7 = torch.cat([in_6, in_7, in_8, conv2d], 1);  in_6 = in_7 = in_8 = None
        tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_1, in_2, in_4, in_3, False, 0.1, 1e-05);  tmp_7 = in_1 = in_2 = in_4 = in_3 = None
        tmp_9 = torch.nn.functional.relu(tmp_8, inplace = True);  tmp_8 = None
        return (conv2d, tmp_9)
        
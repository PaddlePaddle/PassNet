import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_5 = torch.nn.functional.relu(in_6, inplace = True);  in_6 = None
        conv2d = torch.conv2d(tmp_5, in_0, None, (1, 1), (0, 0), (1, 1), 1);  tmp_5 = in_0 = None
        tmp_7 = conv2d + in_5;  conv2d = in_5 = None
        tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_1, in_2, in_4, in_3, False, 0.1, 1e-05);  in_1 = in_2 = in_4 = in_3 = None
        return (tmp_7, tmp_8)
        
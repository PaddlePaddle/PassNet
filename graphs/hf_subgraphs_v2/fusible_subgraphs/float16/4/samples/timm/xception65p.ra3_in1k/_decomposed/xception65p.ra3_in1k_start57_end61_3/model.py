import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        conv2d = torch.conv2d(in_6, in_0, None, (1, 1), (0, 0), (1, 1), 1);  in_6 = in_0 = None
        tmp_6 = conv2d + in_5;  conv2d = in_5 = None
        tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_1, in_2, in_4, in_3, False, 0.1, 0.001);  tmp_6 = in_1 = in_2 = in_4 = in_3 = None
        tmp_8 = torch.nn.functional.relu(tmp_7, inplace = True);  tmp_7 = None
        return (tmp_8,)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        conv2d = torch.conv2d(in_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_4 = in_1 = in_0 = None
        tmp_5 = torch.nn.functional.layer_norm(conv2d, (16, 1, 1), in_3, in_2, 1e-05);  conv2d = in_3 = in_2 = None
        tmp_6 = torch.nn.functional.relu(tmp_5, inplace = True);  tmp_5 = None
        return (tmp_6,)
        
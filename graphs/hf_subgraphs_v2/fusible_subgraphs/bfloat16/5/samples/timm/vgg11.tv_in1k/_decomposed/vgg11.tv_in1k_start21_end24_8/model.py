import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_2 = in_1 = in_0 = None
        tmp_3 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False);  tmp_3 = None
        return (tmp_4,)
        
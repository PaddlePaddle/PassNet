import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        tmp_6 = torch.conv_transpose2d(in_6, in_5, in_4, (2, 2), (1, 1), (1, 1), 1, (1, 1));  in_6 = in_5 = in_4 = None
        tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_0, in_1, in_3, in_2, False, 0.1, 0.001);  tmp_6 = in_0 = in_1 = in_3 = in_2 = None
        tmp_8 = torch.nn.functional.relu(tmp_7, inplace = False);  tmp_7 = None
        return (tmp_8,)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  in_4 = in_0 = in_1 = in_3 = in_2 = None
        tmp_5 = torch.nn.functional.relu(tmp_4, inplace = False);  tmp_4 = None
        return (tmp_5,)
        
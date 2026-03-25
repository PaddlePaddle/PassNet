import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor):
        tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  in_4 = in_0 = in_1 = in_3 = in_2 = None
        tmp_5 = in_5 + tmp_4;  in_5 = tmp_4 = None
        tmp_6 = torch.nn.functional.relu(tmp_5, inplace = False);  tmp_5 = None
        tmp_7 = tmp_6.mean((2, 3), keepdim = True)
        return (tmp_6, tmp_7)
        
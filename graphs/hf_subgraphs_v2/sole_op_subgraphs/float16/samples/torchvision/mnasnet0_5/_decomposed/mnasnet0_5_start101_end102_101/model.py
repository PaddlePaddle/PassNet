import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor):
        tmp_4 = torch.nn.functional.batch_norm(in_0, w_0, w_1, w_3, w_2, False, 0.00029999999999996696, 1e-05);  in_0 = w_0 = w_1 = w_3 = w_2 = None
        return (tmp_4,)
        
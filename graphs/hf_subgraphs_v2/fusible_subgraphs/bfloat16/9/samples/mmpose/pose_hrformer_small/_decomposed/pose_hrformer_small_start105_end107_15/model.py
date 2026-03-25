import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        matmul = in_1 @ in_2;  in_1 = in_2 = None
        tmp_2 = in_0.view(-1);  in_0 = None
        return (matmul, tmp_2)
        
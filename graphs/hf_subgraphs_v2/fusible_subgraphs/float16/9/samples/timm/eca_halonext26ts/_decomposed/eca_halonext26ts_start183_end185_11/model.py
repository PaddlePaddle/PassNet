import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        matmul = in_0 @ in_1;  in_0 = in_1 = None
        tmp_1 = matmul.reshape(-1, 4, 23);  matmul = None
        return (tmp_1,)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        matmul = in_1 @ in_0;  in_1 = in_0 = None
        tmp_1 = matmul.view(1, 384, 20, 20);  matmul = None
        return (tmp_1,)
        
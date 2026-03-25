import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        matmul = in_1 @ in_0;  in_0 = None
        tmp_1 = matmul.permute(0, 3, 1, 2);  matmul = None
        tmp_2 = in_1.permute(0, 3, 1, 2);  in_1 = None
        return (tmp_2, tmp_1)
        
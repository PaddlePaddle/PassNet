import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        matmul = torch.matmul(in_1, in_0);  in_1 = in_0 = None
        tmp_1 = matmul.view(1, 512, 1, 1);  matmul = None
        return (tmp_1,)
        
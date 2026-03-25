import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        matmul = torch.matmul(in_1, in_0);  in_1 = in_0 = None
        tmp_1 = matmul.view(45, 1, 4, 45);  matmul = None
        tmp_2 = tmp_1.permute(1, 2, 0, 3);  tmp_1 = None
        return (tmp_2,)
        
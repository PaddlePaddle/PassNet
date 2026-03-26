import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        matmul = torch.matmul(in_1, in_0);  in_1 = in_0 = None
        tmp_1 = torch.reshape(matmul, [-1, 384]);  matmul = None
        tmp_2 = in_2.transpose(-1, -2);  in_2 = None
        return (tmp_1, tmp_2)
        
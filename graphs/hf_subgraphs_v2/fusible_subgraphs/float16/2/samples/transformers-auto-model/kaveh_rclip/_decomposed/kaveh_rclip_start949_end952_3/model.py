import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        matmul = torch.matmul(in_2, in_1);  in_2 = in_1 = None
        tmp_1 = matmul * in_0;  matmul = in_0 = None
        tmp_2 = tmp_1.T
        return (tmp_1, tmp_2)
        
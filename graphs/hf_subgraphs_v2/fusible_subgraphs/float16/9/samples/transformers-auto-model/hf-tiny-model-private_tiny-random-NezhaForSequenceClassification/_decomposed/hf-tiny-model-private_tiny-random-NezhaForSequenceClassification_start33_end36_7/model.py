import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        matmul = torch.matmul(in_1, in_2);  in_2 = None
        tmp_2 = in_0[(slice(None, 45, None), slice(None, 45, None), slice(None, None, None))];  in_0 = None
        tmp_3 = in_1.permute(2, 0, 1, 3);  in_1 = None
        return (tmp_3, matmul, tmp_2)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        matmul = in_1 @ in_2;  in_2 = None
        tmp_2 = matmul * 0.25;  matmul = None
        tmp_3 = in_1.reshape(-1, 8, 8, 16);  in_1 = None
        tmp_4 = in_0.transpose(-1, -2);  in_0 = None
        return (tmp_2, tmp_3, tmp_4)
        
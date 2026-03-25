import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        matmul = in_0 @ in_1;  in_0 = in_1 = None
        tmp_1 = matmul.transpose(1, 2);  matmul = None
        tmp_2 = tmp_1.reshape(70, 49, 32);  tmp_1 = None
        return (tmp_2,)
        
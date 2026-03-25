import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        einsum = torch.functional.einsum('bnhw,bchw->bnc', in_1, in_2);  in_1 = in_2 = None
        tmp_1 = in_0.reshape(1, 150, 512, -1);  in_0 = None
        tmp_2 = tmp_1.permute(0, 1, 3, 2);  tmp_1 = None
        tmp_3 = einsum.reshape(-1, 256);  einsum = None
        return (tmp_2, tmp_3)
        
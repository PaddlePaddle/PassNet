import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = in_1.view(64, -1, 2, 32);  in_1 = None
        tmp_1 = tmp_0.transpose(1, 2);  tmp_0 = None
        tmp_2 = in_0.permute(0, 2, 1);  in_0 = None
        tmp_3 = tmp_2.reshape(64, 64, 64, 64);  tmp_2 = None
        return (tmp_1, tmp_3)
        
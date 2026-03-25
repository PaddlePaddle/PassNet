import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = in_2.contiguous();  in_2 = None
        tmp_3 = in_1.view((1, 1, 32));  in_1 = None
        tmp_4 = tmp_2.unsqueeze(2)
        tmp_5 = tmp_4.expand((1, 4096, 32, 512));  tmp_4 = None
        tmp_6 = in_0.view((1, 1, 32, 512));  in_0 = None
        return (tmp_5, tmp_6, tmp_3, tmp_2)
        
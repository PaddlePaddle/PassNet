import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_1 = in_1.contiguous();  in_1 = None
        tmp_2 = tmp_1.reshape(24, -1, 64, 64);  tmp_1 = None
        tmp_3 = tmp_2 * in_0;  tmp_2 = in_0 = None
        tmp_4 = tmp_3 + in_2;  tmp_3 = in_2 = None
        return (tmp_4,)
        
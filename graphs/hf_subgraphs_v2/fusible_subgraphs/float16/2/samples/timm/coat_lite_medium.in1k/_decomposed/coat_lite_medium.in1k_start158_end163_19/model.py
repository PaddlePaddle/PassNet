import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = in_0 + in_1;  in_0 = in_1 = None
        tmp_1 = tmp_0[(slice(None, None, None), slice(1, None, None), slice(None, None, None))];  tmp_0 = None
        tmp_2 = tmp_1.reshape(1, 56, 56, -1);  tmp_1 = None
        tmp_3 = tmp_2.permute(0, 3, 1, 2);  tmp_2 = None
        tmp_4 = tmp_3.contiguous();  tmp_3 = None
        return (tmp_4,)
        
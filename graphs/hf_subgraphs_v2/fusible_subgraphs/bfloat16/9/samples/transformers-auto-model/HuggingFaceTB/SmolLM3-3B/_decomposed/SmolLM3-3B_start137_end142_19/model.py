import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_0 = in_1.reshape(1, 16, 2, 128);  in_1 = None
        tmp_1 = in_0[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 2, None))];  in_0 = None
        tmp_2 = in_3.contiguous();  in_3 = None
        tmp_3 = in_2.contiguous();  in_2 = None
        tmp_4 = tmp_0.contiguous();  tmp_0 = None
        return (tmp_1, tmp_3, tmp_2, tmp_4)
        
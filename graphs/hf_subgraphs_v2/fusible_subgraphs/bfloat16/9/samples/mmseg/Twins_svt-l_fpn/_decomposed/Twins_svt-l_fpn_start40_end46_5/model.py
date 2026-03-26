import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.reshape(1, 19, 19, 7, 7, 128);  in_0 = None
        tmp_1 = tmp_0.transpose(2, 3);  tmp_0 = None
        tmp_2 = tmp_1.reshape(1, 133, 133, 128);  tmp_1 = None
        tmp_3 = tmp_2[(slice(None, None, None), slice(None, 128, None), slice(None, 128, None), slice(None, None, None))];  tmp_2 = None
        tmp_4 = tmp_3.contiguous();  tmp_3 = None
        tmp_5 = tmp_4.reshape(1, 16384, 128);  tmp_4 = None
        return (tmp_5,)
        
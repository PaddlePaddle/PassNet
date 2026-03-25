import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = in_3.contiguous();  in_3 = None
        tmp_3 = tmp_2.view(-1, 133, 133, 96);  tmp_2 = None
        tmp_4 = torch.roll(tmp_3, shifts = (3, 3), dims = (1, 2));  tmp_3 = None
        tmp_5 = tmp_4[(slice(None, None, None), slice(None, 128, None), slice(None, 128, None), slice(None, None, None))];  tmp_4 = None
        tmp_6 = tmp_5.contiguous();  tmp_5 = None
        tmp_7 = tmp_6.view(1, 16384, 96);  tmp_6 = None
        tmp_8 = in_2 + tmp_7;  in_2 = tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (96,), in_1, in_0, 1e-05);  in_1 = in_0 = None
        return (tmp_8, tmp_9)
        
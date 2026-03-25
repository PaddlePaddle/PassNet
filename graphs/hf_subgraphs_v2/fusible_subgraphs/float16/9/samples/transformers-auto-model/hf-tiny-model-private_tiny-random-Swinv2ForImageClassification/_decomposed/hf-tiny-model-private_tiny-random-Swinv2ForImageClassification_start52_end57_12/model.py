import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = in_3.contiguous();  in_3 = None
        tmp_3 = tmp_2.view(-1, 16, 16, 16);  tmp_2 = None
        tmp_4 = tmp_3.view(1, 256, 16);  tmp_3 = None
        tmp_5 = torch.nn.functional.layer_norm(tmp_4, (16,), in_1, in_0, 1e-05);  tmp_4 = in_1 = in_0 = None
        tmp_6 = in_2 + tmp_5;  in_2 = tmp_5 = None
        return (tmp_6,)
        
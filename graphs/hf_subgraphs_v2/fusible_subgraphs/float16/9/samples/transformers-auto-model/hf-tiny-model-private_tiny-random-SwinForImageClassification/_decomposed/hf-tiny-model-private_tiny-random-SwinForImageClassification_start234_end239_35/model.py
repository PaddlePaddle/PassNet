import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = in_3.contiguous();  in_3 = None
        tmp_3 = tmp_2.view(-1, 4, 4, 64);  tmp_2 = None
        tmp_4 = tmp_3.view(1, 16, 64);  tmp_3 = None
        tmp_5 = in_2 + tmp_4;  in_2 = tmp_4 = None
        tmp_6 = torch.nn.functional.layer_norm(tmp_5, (64,), in_1, in_0, 1e-05);  in_1 = in_0 = None
        return (tmp_5, tmp_6)
        
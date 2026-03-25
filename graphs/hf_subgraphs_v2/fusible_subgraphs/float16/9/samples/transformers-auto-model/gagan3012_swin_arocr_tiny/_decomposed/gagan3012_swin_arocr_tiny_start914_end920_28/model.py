import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = in_3.contiguous();  in_3 = None
        tmp_3 = tmp_2.view(-1, 32, 32, 768);  tmp_2 = None
        tmp_4 = torch.roll(tmp_3, shifts = (4, 4), dims = (1, 2));  tmp_3 = None
        tmp_5 = tmp_4.view(1, 1024, 768);  tmp_4 = None
        tmp_6 = torch.nn.functional.layer_norm(tmp_5, (768,), in_1, in_0, 1e-05);  tmp_5 = in_1 = in_0 = None
        tmp_7 = in_2 + tmp_6;  in_2 = tmp_6 = None
        return (tmp_7,)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        tmp_2 = in_3[in_4];  in_3 = in_4 = None
        tmp_3 = tmp_2.view(197, 197, -1);  tmp_2 = None
        tmp_4 = tmp_3.permute(2, 0, 1);  tmp_3 = None
        tmp_5 = tmp_4.contiguous();  tmp_4 = None
        tmp_6 = tmp_5.unsqueeze(0);  tmp_5 = None
        tmp_7 = torch.nn.functional.layer_norm(in_2, (768,), in_1, in_0, 1e-12);  in_2 = in_1 = in_0 = None
        return (tmp_7, tmp_6)
        
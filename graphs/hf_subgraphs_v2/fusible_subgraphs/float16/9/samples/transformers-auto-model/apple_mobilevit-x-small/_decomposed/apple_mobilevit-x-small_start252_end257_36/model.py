import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = in_3 + in_2;  in_3 = in_2 = None
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (120,), in_1, in_0, 1e-05);  tmp_2 = in_1 = in_0 = None
        tmp_4 = tmp_3.contiguous();  tmp_3 = None
        tmp_5 = tmp_4.view(1, 4, 64, -1);  tmp_4 = None
        tmp_6 = tmp_5.transpose(1, 3);  tmp_5 = None
        return (tmp_6,)
        
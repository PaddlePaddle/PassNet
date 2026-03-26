import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = in_3 + in_2;  in_3 = in_2 = None
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (64,), in_1, in_0, 1e-05);  tmp_2 = in_1 = in_0 = None
        tmp_4 = tmp_3.reshape(12, 64, 64, -1);  tmp_3 = None
        tmp_5 = tmp_4.permute(0, 3, 1, 2);  tmp_4 = None
        tmp_6 = tmp_5.contiguous();  tmp_5 = None
        return (tmp_6,)
        
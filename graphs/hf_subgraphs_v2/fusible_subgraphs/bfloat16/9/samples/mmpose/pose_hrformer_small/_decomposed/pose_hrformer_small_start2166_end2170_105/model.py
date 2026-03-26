import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = in_3 + in_2;  in_3 = in_2 = None
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (256,), in_1, in_0, 1e-06);  in_1 = in_0 = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = tmp_4.reshape(1, 256, 8, 6);  tmp_4 = None
        return (tmp_2, tmp_5)
        
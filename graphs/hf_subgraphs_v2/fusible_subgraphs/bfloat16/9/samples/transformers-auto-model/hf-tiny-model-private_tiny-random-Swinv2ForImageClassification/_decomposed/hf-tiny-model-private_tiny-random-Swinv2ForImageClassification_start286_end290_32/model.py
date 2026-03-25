import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        tmp_4 = torch.nn.functional.layer_norm(in_5, (64,), in_1, in_0, 1e-05);  in_5 = in_1 = in_0 = None
        tmp_5 = in_4 + tmp_4;  in_4 = tmp_4 = None
        tmp_6 = torch.nn.functional.layer_norm(tmp_5, (64,), in_3, in_2, 1e-05);  tmp_5 = in_3 = in_2 = None
        tmp_7 = tmp_6.transpose(1, 2);  tmp_6 = None
        return (tmp_7,)
        
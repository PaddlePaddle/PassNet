import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_4 = torch.nn.functional.dropout(in_2, p = 0.1, training = False);  in_2 = None
        tmp_5 = in_3 + tmp_4;  in_3 = tmp_4 = None
        tmp_6 = torch.nn.functional.layer_norm(tmp_5, (256,), in_1, in_0, 1e-05);  tmp_5 = in_1 = in_0 = None
        return (tmp_6,)
        
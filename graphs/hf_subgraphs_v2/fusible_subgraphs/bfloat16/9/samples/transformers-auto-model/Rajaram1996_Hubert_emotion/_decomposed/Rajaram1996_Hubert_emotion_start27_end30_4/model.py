import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = torch.nn.functional.layer_norm(in_2, (768,), in_1, in_0, 1e-05);  in_2 = in_1 = in_0 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False);  tmp_2 = None
        tmp_4 = torch.rand([]);  tmp_4 = None
        return (tmp_3,)
        
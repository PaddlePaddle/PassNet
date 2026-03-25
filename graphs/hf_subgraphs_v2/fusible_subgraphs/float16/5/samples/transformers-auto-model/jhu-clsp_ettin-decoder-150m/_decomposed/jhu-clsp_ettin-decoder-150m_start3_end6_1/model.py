import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_3 = torch.nn.functional.embedding(in_0, in_2, 50283, None, 2.0, False, False);  in_0 = in_2 = None
        tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, None, 1e-05);  tmp_3 = in_1 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.0, False, False);  tmp_4 = None
        return (tmp_5,)
        
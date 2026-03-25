import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_4 = torch.nn.functional.embedding(in_4, in_1, None, None, 2.0, False, False);  in_4 = in_1 = None
        tmp_5 = in_0 + tmp_4;  in_0 = tmp_4 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, p = 0.1, training = False);  tmp_5 = None
        tmp_7 = torch.nn.functional.layer_norm(tmp_6, (1024,), in_3, in_2, 1e-05);  in_3 = in_2 = None
        return (tmp_6, tmp_7)
        
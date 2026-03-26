import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor):
        tmp_6 = torch.nn.functional.embedding(in_0, in_4, 0, None, 2.0, False, False);  in_0 = in_4 = None
        tmp_7 = torch.nn.functional.embedding(in_5, in_3, None, None, 2.0, False, False);  in_5 = in_3 = None
        tmp_8 = tmp_6 + tmp_7;  tmp_6 = tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (32,), in_2, in_1, 1e-12);  tmp_8 = in_2 = in_1 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.1, False, False);  tmp_9 = None
        return (tmp_10,)
        
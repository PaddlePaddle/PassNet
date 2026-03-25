import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor):
        tmp_7 = torch.nn.functional.embedding(in_0, in_5, 0, None, 2.0, False, False);  in_0 = in_5 = None
        tmp_8 = torch.nn.functional.embedding(in_6, in_4, None, None, 2.0, False, False);  in_6 = in_4 = None
        tmp_9 = tmp_7 + tmp_8;  tmp_7 = tmp_8 = None
        tmp_10 = torch.nn.functional.embedding(in_7, in_3, None, None, 2.0, False, False);  in_7 = in_3 = None
        tmp_9 += tmp_10;  tmp_11 = tmp_9;  tmp_9 = tmp_10 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (32,), in_2, in_1, 1e-05);  tmp_11 = in_2 = in_1 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.1, False, False);  tmp_12 = None
        return (tmp_13,)
        
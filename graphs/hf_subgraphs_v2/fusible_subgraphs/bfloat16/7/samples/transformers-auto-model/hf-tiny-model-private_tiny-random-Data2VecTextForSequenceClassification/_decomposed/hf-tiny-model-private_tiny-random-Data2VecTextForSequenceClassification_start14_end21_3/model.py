import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor):
        tmp_6 = torch.nn.functional.embedding(in_0, in_5, 1, None, 2.0, False, False);  in_0 = in_5 = None
        tmp_7 = torch.nn.functional.embedding(in_6, in_4, None, None, 2.0, False, False);  in_6 = in_4 = None
        tmp_8 = tmp_6 + tmp_7;  tmp_6 = tmp_7 = None
        tmp_9 = torch.nn.functional.embedding(in_7, in_3, 1, None, 2.0, False, False);  in_7 = in_3 = None
        tmp_8 += tmp_9;  tmp_10 = tmp_8;  tmp_8 = tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (32,), in_2, in_1, 1e-12);  tmp_10 = in_2 = in_1 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.1, False, False);  tmp_11 = None
        return (tmp_12,)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor):
        tmp_8 = torch.nn.functional.embedding(in_0, in_5, 1, None, 2.0, False, False);  in_0 = in_5 = None
        tmp_9 = torch.nn.functional.embedding(in_8, in_4, None, None, 2.0, False, False);  in_8 = in_4 = None
        tmp_10 = tmp_8 + tmp_9;  tmp_8 = tmp_9 = None
        tmp_11 = torch.nn.functional.embedding(in_9, in_3, 1, None, 2.0, False, False);  in_9 = in_3 = None
        tmp_10 += tmp_11;  tmp_12 = tmp_10;  tmp_10 = tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (1024,), in_2, in_1, 1e-05);  tmp_12 = in_2 = in_1 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.1, False, False);  tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (1024,), in_7, in_6, 1e-05);  in_7 = in_6 = None
        return (tmp_14, tmp_15)
        
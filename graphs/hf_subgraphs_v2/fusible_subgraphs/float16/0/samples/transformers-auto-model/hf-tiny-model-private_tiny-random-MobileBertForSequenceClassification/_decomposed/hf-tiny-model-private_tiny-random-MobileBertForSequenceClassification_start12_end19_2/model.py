import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        tmp_5 = torch.nn.functional.embedding(in_5, in_2, None, None, 2.0, False, False);  in_5 = in_2 = None
        tmp_6 = torch.nn.functional.embedding(in_4, in_3, None, None, 2.0, False, False);  in_4 = in_3 = None
        tmp_7 = in_6 + tmp_5;  in_6 = tmp_5 = None
        tmp_8 = tmp_7 + tmp_6;  tmp_7 = tmp_6 = None
        tmp_9 = tmp_8 * in_1;  tmp_8 = in_1 = None
        tmp_10 = tmp_9 + in_0;  tmp_9 = in_0 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.1, False, False);  tmp_10 = None
        return (tmp_11,)
        
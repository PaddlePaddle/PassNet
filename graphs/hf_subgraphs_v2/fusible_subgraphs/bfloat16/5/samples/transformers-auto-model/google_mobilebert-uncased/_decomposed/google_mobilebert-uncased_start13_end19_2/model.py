import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor):
        tmp_4 = torch.nn.functional.embedding(in_3, in_2, None, None, 2.0, False, False);  in_3 = in_2 = None
        tmp_5 = in_4 + in_5;  in_4 = in_5 = None
        tmp_6 = tmp_5 + tmp_4;  tmp_5 = tmp_4 = None
        tmp_7 = tmp_6 * in_1;  tmp_6 = in_1 = None
        tmp_8 = tmp_7 + in_0;  tmp_7 = in_0 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False);  tmp_8 = None
        return (tmp_9,)
        
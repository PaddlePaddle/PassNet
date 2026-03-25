import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor):
        tmp_5 = torch.nn.functional.embedding(in_0, in_3, 0, None, 2.0, False, False);  in_0 = in_3 = None
        tmp_6 = torch.nn.functional.embedding(in_4, in_2, None, None, 2.0, False, False);  in_4 = in_2 = None
        tmp_7 = tmp_5 + tmp_6;  tmp_5 = tmp_6 = None
        tmp_8 = torch.nn.functional.embedding(in_5, in_1, None, None, 2.0, False, False);  in_5 = in_1 = None
        tmp_7 += tmp_8;  tmp_9 = tmp_7;  tmp_7 = tmp_8 = None
        return (tmp_9,)
        
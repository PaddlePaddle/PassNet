import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = in_0.ne(1);  in_0 = None
        tmp_2 = tmp_1.int();  tmp_1 = None
        tmp_3 = torch.cumsum(tmp_2, dim = 1)
        tmp_4 = tmp_3.type_as(tmp_2);  tmp_3 = None
        tmp_5 = tmp_4 + 0;  tmp_4 = None
        tmp_6 = tmp_5 * tmp_2;  tmp_5 = tmp_2 = None
        tmp_7 = tmp_6.long();  tmp_6 = None
        tmp_8 = tmp_7 + 1;  tmp_7 = None
        return (tmp_8,)
        
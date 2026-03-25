import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = in_1 + in_0;  in_1 = in_0 = None
        tmp_1 = tmp_0.view(8, 625, 625);  tmp_0 = None
        tmp_2 = torch.nn.functional.softmax(tmp_1, dim = -1);  tmp_1 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, p = 0.0, training = False);  tmp_2 = None
        return (tmp_3,)
        
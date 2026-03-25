import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.softmax(in_0, dim = -1);  in_0 = None
        tmp_1 = tmp_0.view(1, 8, 300, 300);  tmp_0 = None
        tmp_2 = tmp_1.view(8, 300, 300)
        tmp_3 = torch.nn.functional.dropout(tmp_2, p = 0.0, training = False);  tmp_2 = None
        return (tmp_3, tmp_1)
        
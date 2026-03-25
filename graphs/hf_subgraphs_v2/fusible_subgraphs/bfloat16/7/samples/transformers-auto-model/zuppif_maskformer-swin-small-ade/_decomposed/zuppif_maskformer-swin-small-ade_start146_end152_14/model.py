import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.pad(in_0, (0, 0, 0, 6, 0, 6), 'constant', None);  in_0 = None
        tmp_1 = tmp_0.view(1, 10, 7, 10, 7, 192);  tmp_0 = None
        tmp_2 = tmp_1.permute(0, 1, 3, 2, 4, 5);  tmp_1 = None
        tmp_3 = tmp_2.contiguous();  tmp_2 = None
        tmp_4 = tmp_3.view(-1, 7, 7, 192);  tmp_3 = None
        tmp_5 = tmp_4.view(-1, 49, 192);  tmp_4 = None
        return (tmp_5,)
        
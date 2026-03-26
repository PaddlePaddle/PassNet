import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_1 = torch.nn.functional.silu(in_1, inplace = True);  in_1 = None
        tmp_2 = tmp_1.mean((2, 3))
        tmp_3 = tmp_2.view(1, 1, -1);  tmp_2 = None
        conv1d = torch.conv1d(tmp_3, in_0, None, (1,), (1,), (1,), 1);  tmp_3 = in_0 = None
        tmp_5 = conv1d.sigmoid();  conv1d = None
        tmp_6 = tmp_5.view(1, -1, 1, 1);  tmp_5 = None
        tmp_7 = tmp_6.expand_as(tmp_1);  tmp_6 = None
        tmp_8 = tmp_1 * tmp_7;  tmp_1 = tmp_7 = None
        return (tmp_8,)
        
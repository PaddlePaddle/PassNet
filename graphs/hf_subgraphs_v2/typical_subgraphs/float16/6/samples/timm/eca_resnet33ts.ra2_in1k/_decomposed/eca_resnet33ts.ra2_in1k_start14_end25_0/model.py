import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_1 = torch.nn.functional.silu(in_2, inplace = True);  in_2 = None
        tmp_2 = tmp_1.mean((2, 3))
        tmp_3 = torch.sym_sum([-1, in_1]);  in_1 = None
        tmp_4 = tmp_3 // 4
        tmp_5 = torch.sym_sum([1, tmp_4]);  tmp_4 = tmp_5 = None
        tmp_6 = tmp_2.view(1, 1, -1);  tmp_2 = None
        conv1d = torch.conv1d(tmp_6, in_0, None, (1,), (1,), (1,), 1);  tmp_6 = in_0 = None
        tmp_8 = conv1d.sigmoid();  conv1d = None
        tmp_9 = tmp_8.view(1, -1, 1, 1);  tmp_8 = None
        tmp_10 = tmp_9.expand_as(tmp_1);  tmp_9 = None
        tmp_11 = tmp_1 * tmp_10;  tmp_1 = tmp_10 = None
        return (tmp_3, tmp_11)
        
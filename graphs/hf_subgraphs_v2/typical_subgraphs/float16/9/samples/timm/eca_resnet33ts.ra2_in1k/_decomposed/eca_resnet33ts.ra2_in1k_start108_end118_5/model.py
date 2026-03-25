import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0, in_1):
        tmp_1 = torch.nn.functional.silu(in_1, inplace = True);  in_1 = None
        tmp_2 = tmp_1.mean((2, 3))
        tmp_3 = in_0 // 16;  in_0 = None
        tmp_4 = torch.sym_sum([1, tmp_3]);  tmp_3 = tmp_4 = None
        tmp_5 = tmp_2.view(1, 1, -1);  tmp_2 = None
        conv1d = torch.conv1d(tmp_5, w_0, None, (1,), (2,), (1,), 1);  tmp_5 = w_0 = None
        tmp_7 = conv1d.sigmoid();  conv1d = None
        tmp_8 = tmp_7.view(1, -1, 1, 1);  tmp_7 = None
        tmp_9 = tmp_8.expand_as(tmp_1);  tmp_8 = None
        tmp_10 = tmp_1 * tmp_9;  tmp_1 = tmp_9 = None
        return (tmp_10,)
        
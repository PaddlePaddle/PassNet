import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        tmp_1 = torch.nn.functional.avg_pool2d(tmp_0, 2, 1, 0, False, True)
        chunk = tmp_1.chunk(2, 1);  tmp_1 = None
        tmp_3 = chunk[0]
        tmp_4 = chunk[1];  chunk = None
        return (tmp_3, tmp_4, tmp_0)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_4 = torch.nn.functional.avg_pool2d(in_4, 2, 2, 0, False, True, None)
        tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  in_4 = in_0 = in_1 = in_3 = in_2 = None
        tmp_6 = torch.nn.functional.silu(tmp_5, inplace = True);  tmp_5 = None
        return (tmp_4, tmp_6)
        
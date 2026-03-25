import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_4 = in_4.reshape(1, 512, 16, 16);  in_4 = None
        tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None);  tmp_4 = None
        tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_5 = in_0 = in_1 = in_3 = in_2 = None
        tmp_7 = torch.nn.functional.silu(tmp_6, inplace = True);  tmp_6 = None
        return (tmp_7,)
        
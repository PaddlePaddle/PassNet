import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        tmp_4 = in_4.reshape(-1, 8, 8, 1, 1);  in_4 = None
        tmp_5 = tmp_4.permute(0, 3, 1, 4, 2);  tmp_4 = None
        tmp_6 = tmp_5.contiguous();  tmp_5 = None
        tmp_7 = tmp_6.view(1, 512, 8, 8);  tmp_6 = None
        tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_7 = in_0 = in_1 = in_3 = in_2 = None
        tmp_9 = torch.nn.functional.silu(tmp_8, inplace = True);  tmp_8 = None
        return (tmp_9,)
        
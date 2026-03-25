import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor):
        in_5 += in_6;  in_8 = in_5;  in_5 = in_6 = None
        tmp_6 = torch.nn.functional.batch_norm(in_7, in_1, in_2, in_4, in_3, False, 0.1, 1e-05);  in_1 = in_2 = in_4 = in_3 = None
        tmp_7 = 0 + tmp_6;  tmp_6 = None
        tmp_8 = in_8 - tmp_7;  in_8 = tmp_7 = None
        tmp_9 = tmp_8 * in_0;  tmp_8 = in_0 = None
        tmp_10 = in_7 + tmp_9;  in_7 = tmp_9 = None
        return (tmp_10,)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        in_0 += in_1;  in_3 = in_0;  in_0 = in_1 = None
        tmp_6 = torch.nn.functional.batch_norm(in_2, w_1, w_2, w_4, w_3, False, 0.1, 1e-05);  w_1 = w_2 = w_4 = w_3 = None
        tmp_7 = 0 + tmp_6;  tmp_6 = None
        tmp_8 = in_3 - tmp_7;  in_3 = tmp_7 = None
        tmp_9 = tmp_8 * w_0;  tmp_8 = w_0 = None
        tmp_10 = in_2 + tmp_9;  in_2 = tmp_9 = None
        return (tmp_10,)
        
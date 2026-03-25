import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = in_0 + in_2;  in_0 = in_2 = None
        tmp_3 = tmp_2.to(torch.float32)
        tmp_4 = tmp_3.pow(2)
        tmp_5 = tmp_4.mean(-1, keepdim = True);  tmp_4 = None
        tmp_6 = tmp_5 + 1e-06;  tmp_5 = None
        tmp_7 = torch.rsqrt(tmp_6);  tmp_6 = None
        tmp_8 = tmp_3 * tmp_7;  tmp_3 = tmp_7 = None
        tmp_9 = tmp_8.to(torch.bfloat16);  tmp_8 = None
        tmp_10 = in_1 * tmp_9;  in_1 = tmp_9 = None
        return (tmp_2, tmp_10)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_2 = in_0.to(torch.float32);  in_0 = None
        tmp_3 = tmp_2.pow(2)
        tmp_4 = tmp_3.mean(-1, keepdim = True);  tmp_3 = None
        tmp_5 = tmp_4 + 1e-05;  tmp_4 = None
        tmp_6 = torch.rsqrt(tmp_5);  tmp_5 = None
        tmp_7 = tmp_2 * tmp_6;  tmp_2 = tmp_6 = None
        tmp_8 = tmp_7.to(torch.bfloat16);  tmp_7 = None
        tmp_9 = in_1 * tmp_8;  in_1 = tmp_8 = None
        return (tmp_9,)
        
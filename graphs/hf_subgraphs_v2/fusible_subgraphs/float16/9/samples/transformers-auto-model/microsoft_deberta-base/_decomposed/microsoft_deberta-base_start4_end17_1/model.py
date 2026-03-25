import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_3 = in_3 + in_2;  in_3 = in_2 = None
        tmp_4 = tmp_3.float();  tmp_3 = None
        tmp_5 = tmp_4.mean(-1, keepdim = True)
        tmp_6 = tmp_4 - tmp_5
        tmp_7 = tmp_6.pow(2);  tmp_6 = None
        tmp_8 = tmp_7.mean(-1, keepdim = True);  tmp_7 = None
        tmp_9 = tmp_4 - tmp_5;  tmp_4 = tmp_5 = None
        tmp_10 = tmp_8 + 1e-07;  tmp_8 = None
        tmp_11 = torch.sqrt(tmp_10);  tmp_10 = None
        tmp_12 = tmp_9 / tmp_11;  tmp_9 = tmp_11 = None
        tmp_13 = tmp_12.to(torch.float32);  tmp_12 = None
        tmp_14 = in_1 * tmp_13;  in_1 = tmp_13 = None
        tmp_15 = tmp_14 + in_0;  tmp_14 = in_0 = None
        return (tmp_15,)
        
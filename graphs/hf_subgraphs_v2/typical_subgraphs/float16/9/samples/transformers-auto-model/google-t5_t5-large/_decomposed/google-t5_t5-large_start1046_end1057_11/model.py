import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0, in_1, in_2):
        tmp_1 = in_0.view(1, -1, 1024);  in_0 = None
        linear = torch.nn.functional.linear(tmp_1, w_0, None);  tmp_1 = w_0 = None
        tmp_3 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_4 = in_2 + tmp_3;  in_2 = tmp_3 = None
        tmp_5 = in_1[-1];  in_1 = None
        tmp_6 = tmp_5 + 1;  tmp_5 = tmp_6 = None
        tmp_7 = tmp_4.to(torch.float32)
        tmp_8 = tmp_7.pow(2);  tmp_7 = None
        tmp_9 = tmp_8.mean(-1, keepdim = True);  tmp_8 = None
        tmp_10 = tmp_9 + 1e-06;  tmp_9 = None
        tmp_11 = torch.rsqrt(tmp_10);  tmp_10 = None
        return (tmp_4, tmp_11)
        
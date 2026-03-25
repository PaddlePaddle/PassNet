import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, in_0 : torch.Tensor):
        linear = torch.nn.functional.linear(in_0, w_1, w_0);  w_1 = w_0 = None
        tmp_8 = torch.nn.functional.gelu(linear);  linear = None
        linear_1 = torch.nn.functional.linear(tmp_8, w_5, w_4);  tmp_8 = w_5 = w_4 = None
        tmp_10 = torch.nn.functional.dropout(linear_1, 0.1, False, False);  linear_1 = None
        tmp_11 = tmp_10 + in_0;  tmp_10 = in_0 = None
        tmp_12 = tmp_11.float();  tmp_11 = None
        tmp_13 = tmp_12.mean(-1, keepdim = True)
        tmp_14 = tmp_12 - tmp_13
        tmp_15 = tmp_14.pow(2);  tmp_14 = None
        tmp_16 = tmp_15.mean(-1, keepdim = True);  tmp_15 = None
        tmp_17 = tmp_12 - tmp_13;  tmp_12 = tmp_13 = None
        tmp_18 = tmp_16 + 1e-07;  tmp_16 = None
        tmp_19 = torch.sqrt(tmp_18);  tmp_18 = None
        tmp_20 = tmp_17 / tmp_19;  tmp_17 = tmp_19 = None
        tmp_21 = tmp_20.to(torch.float32);  tmp_20 = None
        tmp_22 = w_3 * tmp_21;  w_3 = tmp_21 = None
        tmp_23 = tmp_22 + w_2;  tmp_22 = w_2 = None
        return (tmp_23,)
        
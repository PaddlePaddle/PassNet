import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_4 = in_0.view(1, -1, 1536);  in_0 = None
        linear = torch.nn.functional.linear(tmp_4, w_0, None);  tmp_4 = w_0 = None
        tmp_6 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_7 = in_1 + tmp_6;  in_1 = tmp_6 = None
        tmp_8 = tmp_7.to(torch.float32)
        tmp_9 = tmp_8.pow(2);  tmp_8 = None
        tmp_10 = tmp_9.mean(-1, keepdim = True);  tmp_9 = None
        tmp_11 = tmp_10 + 1e-06;  tmp_10 = None
        tmp_12 = torch.rsqrt(tmp_11);  tmp_11 = None
        tmp_13 = tmp_7 * tmp_12;  tmp_12 = None
        tmp_14 = w_3 * tmp_13;  w_3 = tmp_13 = None
        to_1 = tmp_14.to(torch.bfloat16);  tmp_14 = None
        linear_1 = torch.nn.functional.linear(to_1, w_1, None);  to_1 = w_1 = None
        tmp_16 = torch.nn.functional.relu(linear_1, inplace = False);  linear_1 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.1, False, False);  tmp_16 = None
        linear_2 = torch.nn.functional.linear(tmp_17, w_2, None);  tmp_17 = w_2 = None
        tmp_19 = torch.nn.functional.dropout(linear_2, 0.1, False, False);  linear_2 = None
        tmp_20 = tmp_7 + tmp_19;  tmp_7 = tmp_19 = None
        tmp_21 = tmp_20.to(torch.float32)
        tmp_22 = tmp_21.pow(2);  tmp_21 = None
        tmp_23 = tmp_22.mean(-1, keepdim = True);  tmp_22 = None
        tmp_24 = tmp_23 + 1e-06;  tmp_23 = None
        tmp_25 = torch.rsqrt(tmp_24);  tmp_24 = None
        return (tmp_20, tmp_25)
        
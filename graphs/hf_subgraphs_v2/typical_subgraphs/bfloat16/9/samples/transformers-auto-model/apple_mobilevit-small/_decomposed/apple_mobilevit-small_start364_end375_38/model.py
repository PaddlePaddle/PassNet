import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, in_0, in_1):
        tmp_4 = torch.nn.functional.silu(in_1, inplace = False);  in_1 = None
        linear = torch.nn.functional.linear(tmp_4, w_3, w_2);  tmp_4 = w_3 = w_2 = None
        tmp_6 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_7 = tmp_6 + in_0;  tmp_6 = in_0 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (240,), w_1, w_0, 1e-05);  tmp_7 = w_1 = w_0 = None
        tmp_9 = tmp_8.contiguous();  tmp_8 = None
        tmp_10 = tmp_9.view(1, 4, 16, -1);  tmp_9 = None
        tmp_11 = tmp_10.transpose(1, 3);  tmp_10 = None
        tmp_12 = tmp_11.reshape(960, 4, 2, 2);  tmp_11 = None
        tmp_13 = tmp_12.transpose(1, 2);  tmp_12 = None
        tmp_14 = tmp_13.reshape(1, 240, 8, 8);  tmp_13 = None
        return (tmp_14,)
        
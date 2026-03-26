import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_5 = in_6 + in_5;  in_6 = in_5 = None
        tmp_6 = torch.nn.functional.layer_norm(tmp_5, (256,), in_2, in_1, 1e-12);  tmp_5 = in_2 = in_1 = None
        tmp_7 = tmp_6[(slice(None, None, None), 0)]
        to = tmp_7.to(torch.bfloat16);  tmp_7 = None
        linear = torch.nn.functional.linear(to, in_4, in_3);  to = in_4 = in_3 = None
        tmp_9 = torch.tanh(linear);  linear = tmp_9 = None
        tmp_10 = in_0.unsqueeze(-1);  in_0 = None
        tmp_11 = tmp_10.expand((1, 10, 256));  tmp_10 = None
        tmp_12 = tmp_11.to(torch.float32);  tmp_11 = None
        tmp_13 = tmp_6 * tmp_12
        return (tmp_6, tmp_12, tmp_13)
        
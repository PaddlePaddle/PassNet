import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        tmp_4 = in_5 + in_4;  in_5 = in_4 = None
        tmp_5 = torch.nn.functional.layer_norm(tmp_4, (512,), in_1, in_0, 1e-12);  tmp_4 = in_1 = in_0 = None
        tmp_6 = tmp_5[(slice(None, None, None), 0)]
        to = tmp_6.to(torch.bfloat16);  tmp_6 = None
        linear = torch.nn.functional.linear(to, in_3, in_2);  to = in_3 = in_2 = None
        tmp_8 = torch.tanh(linear);  linear = tmp_8 = None
        tmp_9 = tmp_5[(slice(None, None, None), 0)]
        return (tmp_9, tmp_5)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor):
        tmp_4 = torch.nn.functional.gelu(in_0);  in_0 = None
        tmp_5 = tmp_4.reshape(1, 124, 2, 768);  tmp_4 = None
        tmp_6 = tmp_5.reshape(1, 248, 768);  tmp_5 = None
        tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 0, 1), 'constant', None);  tmp_6 = None
        linear = torch.nn.functional.linear(tmp_7, w_3, w_2);  tmp_7 = w_3 = w_2 = None
        tmp_9 = linear.mean(dim = 1);  linear = None
        linear_1 = torch.nn.functional.linear(tmp_9, w_1, w_0);  tmp_9 = w_1 = w_0 = None
        return (linear_1,)
        
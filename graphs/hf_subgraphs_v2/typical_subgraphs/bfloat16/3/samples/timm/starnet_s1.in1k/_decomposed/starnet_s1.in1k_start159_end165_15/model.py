import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor):
        conv2d = torch.conv2d(in_8, in_7, in_6, (1, 1), (3, 3), (1, 1), 192);  in_8 = in_7 = in_6 = None
        tmp_9 = in_9 + conv2d;  in_9 = conv2d = None
        tmp_10 = torch.nn.functional.batch_norm(tmp_9, in_2, in_3, in_5, in_4, False, 0.1, 1e-05);  tmp_9 = in_2 = in_3 = in_5 = in_4 = None
        tmp_11 = torch.nn.functional.adaptive_avg_pool2d(tmp_10, 1);  tmp_10 = None
        tmp_12 = tmp_11.flatten(1, -1);  tmp_11 = None
        to = tmp_12.to(torch.bfloat16);  tmp_12 = None
        linear = torch.nn.functional.linear(to, in_1, in_0);  to = in_1 = in_0 = None
        return (linear,)
        
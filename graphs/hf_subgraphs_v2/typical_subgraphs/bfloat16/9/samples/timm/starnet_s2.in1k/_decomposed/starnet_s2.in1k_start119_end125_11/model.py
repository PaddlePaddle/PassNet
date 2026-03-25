import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_7, w_6, (1, 1), (3, 3), (1, 1), 256);  in_0 = w_7 = w_6 = None
        tmp_9 = in_1 + conv2d;  in_1 = conv2d = None
        tmp_10 = torch.nn.functional.batch_norm(tmp_9, w_2, w_3, w_5, w_4, False, 0.1, 1e-05);  tmp_9 = w_2 = w_3 = w_5 = w_4 = None
        tmp_11 = torch.nn.functional.adaptive_avg_pool2d(tmp_10, 1);  tmp_10 = None
        tmp_12 = tmp_11.flatten(1, -1);  tmp_11 = None
        linear = torch.nn.functional.linear(tmp_12, w_1, w_0);  tmp_12 = w_1 = w_0 = None
        return (linear,)
        
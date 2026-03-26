import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor):
        tmp_4 = torch.nn.functional.gelu(in_4, approximate = 'none');  in_4 = None
        conv2d = torch.conv2d(tmp_4, in_3, None, (1, 1), (0, 0), (1, 1), 1);  tmp_4 = in_3 = None
        tmp_6 = in_5 + conv2d;  in_5 = conv2d = None
        tmp_7 = torch.nn.functional.adaptive_avg_pool2d(tmp_6, 1);  tmp_6 = None
        conv2d_1 = torch.conv2d(tmp_7, in_2, None, (1, 1), (0, 0), (1, 1), 1);  tmp_7 = in_2 = None
        tmp_9 = torch.nn.functional.gelu(conv2d_1, approximate = 'none');  conv2d_1 = None
        tmp_10 = tmp_9.flatten(1, -1);  tmp_9 = None
        linear = torch.nn.functional.linear(tmp_10, in_1, in_0);  tmp_10 = in_1 = in_0 = None
        return (linear,)
        
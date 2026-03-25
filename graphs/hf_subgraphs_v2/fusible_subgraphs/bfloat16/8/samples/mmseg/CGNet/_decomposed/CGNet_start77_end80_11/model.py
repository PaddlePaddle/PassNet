import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1);  in_1 = in_0 = None
        tmp_2 = torch.nn.functional.adaptive_avg_pool2d(conv2d, 1)
        tmp_3 = tmp_2.view(128, 128);  tmp_2 = None
        return (conv2d, tmp_3)
        
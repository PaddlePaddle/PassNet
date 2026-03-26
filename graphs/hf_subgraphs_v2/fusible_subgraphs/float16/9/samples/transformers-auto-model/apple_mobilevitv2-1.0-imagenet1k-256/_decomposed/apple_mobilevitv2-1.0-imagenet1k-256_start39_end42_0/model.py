import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1);  in_1 = in_0 = None
        tmp_2 = torch.nn.functional.unfold(conv2d, kernel_size = (2, 2), stride = (2, 2));  conv2d = None
        tmp_3 = tmp_2.reshape(1, 128, 4, -1);  tmp_2 = None
        return (tmp_3,)
        
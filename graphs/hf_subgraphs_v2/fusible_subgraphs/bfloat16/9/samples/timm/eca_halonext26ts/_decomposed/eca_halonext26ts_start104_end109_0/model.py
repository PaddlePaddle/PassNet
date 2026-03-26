import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1);  in_1 = in_0 = None
        tmp_2 = conv2d.reshape(-1, 16, 2, 8, 2, 8);  conv2d = None
        tmp_3 = tmp_2.permute(0, 1, 3, 5, 2, 4);  tmp_2 = None
        tmp_4 = tmp_3.reshape(8, 16, -1, 4);  tmp_3 = None
        tmp_5 = tmp_4.transpose(1, 3);  tmp_4 = None
        return (tmp_5,)
        
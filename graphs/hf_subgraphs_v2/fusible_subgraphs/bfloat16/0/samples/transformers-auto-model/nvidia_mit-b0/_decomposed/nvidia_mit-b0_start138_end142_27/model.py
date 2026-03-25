import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        conv2d = torch.conv2d(in_4, in_3, in_2, (4, 4), (0, 0), (1, 1), 1);  in_4 = in_3 = in_2 = None
        tmp_5 = conv2d.reshape(1, 64, -1);  conv2d = None
        tmp_6 = tmp_5.permute(0, 2, 1);  tmp_5 = None
        tmp_7 = torch.nn.functional.layer_norm(tmp_6, (64,), in_1, in_0, 1e-05);  tmp_6 = in_1 = in_0 = None
        return (tmp_7,)
        
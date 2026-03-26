import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        conv2d = torch.conv2d(in_6, in_5, in_4, (2, 2), (1, 1), (1, 1), 1);  in_6 = in_5 = in_4 = None
        tmp_7 = conv2d.flatten(2);  conv2d = None
        tmp_8 = tmp_7.transpose(1, 2);  tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (64,), in_3, in_2, 1e-05);  tmp_8 = in_3 = in_2 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (64,), in_1, in_0, 1e-05);  in_1 = in_0 = None
        return (tmp_9, tmp_10)
        
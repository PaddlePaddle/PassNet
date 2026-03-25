import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        conv2d = torch.conv2d(in_0, in_6, in_5, (4, 4), (3, 3), (1, 1), 1);  in_0 = in_6 = in_5 = None
        tmp_8 = conv2d.flatten(2);  conv2d = None
        tmp_9 = tmp_8.transpose(1, 2);  tmp_8 = None
        tmp_10 = torch.nn.functional.layer_norm(tmp_9, (32,), in_4, in_3, 1e-05);  tmp_9 = in_4 = in_3 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (32,), in_2, in_1, 1e-05);  in_2 = in_1 = None
        return (tmp_10, tmp_11)
        
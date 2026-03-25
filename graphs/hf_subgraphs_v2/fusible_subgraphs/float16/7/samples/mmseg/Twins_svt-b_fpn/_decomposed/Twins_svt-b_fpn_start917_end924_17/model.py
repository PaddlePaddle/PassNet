import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        conv2d = torch.conv2d(in_4, in_3, in_2, (1, 1), (1, 1), (1, 1), 768);  in_3 = in_2 = None
        tmp_5 = conv2d + in_4;  conv2d = in_4 = None
        tmp_6 = tmp_5.flatten(2);  tmp_5 = None
        tmp_7 = tmp_6.transpose(1, 2);  tmp_6 = None
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), in_1, in_0, 1e-05);  in_1 = in_0 = None
        tmp_9 = tmp_8.transpose(0, 1)
        tmp_10 = tmp_8.transpose(0, 1);  tmp_8 = None
        return (tmp_7, tmp_10, tmp_9)
        
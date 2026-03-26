import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        conv1d = torch.conv1d(in_4, in_1, in_0, (2,), (0,), (1,), 1);  in_4 = in_1 = in_0 = None
        tmp_5 = conv1d.transpose(-2, -1);  conv1d = None
        tmp_6 = torch.nn.functional.layer_norm(tmp_5, (512,), in_3, in_2, 1e-05);  tmp_5 = in_3 = in_2 = None
        tmp_7 = tmp_6.transpose(-2, -1);  tmp_6 = None
        return (tmp_7,)
        
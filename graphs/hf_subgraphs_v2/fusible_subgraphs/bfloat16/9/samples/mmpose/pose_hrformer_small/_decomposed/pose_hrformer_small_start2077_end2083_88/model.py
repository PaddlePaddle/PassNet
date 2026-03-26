import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = torch.nn.functional.layer_norm(in_2, (256,), in_1, in_0, 1e-06);  in_2 = in_1 = in_0 = None
        tmp_3 = tmp_2.view(1, 8, 6, 256);  tmp_2 = None
        tmp_4 = torch.nn.functional.pad(tmp_3, (0, 0, 0, 1, 3, 3), 'constant', None);  tmp_3 = None
        tmp_5 = tmp_4.view(1, 2, 7, 1, 7, 256);  tmp_4 = None
        tmp_6 = tmp_5.permute(0, 1, 3, 2, 4, 5);  tmp_5 = None
        tmp_7 = tmp_6.reshape(-1, 49, 256);  tmp_6 = None
        return (tmp_7,)
        
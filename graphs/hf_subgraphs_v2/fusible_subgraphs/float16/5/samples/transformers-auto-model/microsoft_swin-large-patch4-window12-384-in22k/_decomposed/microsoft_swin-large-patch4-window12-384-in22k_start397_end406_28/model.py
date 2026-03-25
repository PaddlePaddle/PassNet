import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_2 = in_2 + in_3;  in_2 = in_3 = None
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), in_1, in_0, 1e-05);  in_1 = in_0 = None
        tmp_4 = tmp_3.view(1, 24, 24, 768);  tmp_3 = None
        tmp_5 = torch.nn.functional.pad(tmp_4, (0, 0, 0, 0, 0, 0), 'constant', None);  tmp_4 = None
        tmp_6 = tmp_5.view(1, 2, 12, 2, 12, 768);  tmp_5 = None
        tmp_7 = tmp_6.permute(0, 1, 3, 2, 4, 5);  tmp_6 = None
        tmp_8 = tmp_7.contiguous();  tmp_7 = None
        tmp_9 = tmp_8.view(-1, 12, 12, 768);  tmp_8 = None
        tmp_10 = tmp_9.view(-1, 144, 768);  tmp_9 = None
        return (tmp_10, tmp_2)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_0 = in_0 + in_1;  in_0 = in_1 = None
        tmp_1 = in_2 * 0.1767766952966369;  in_2 = None
        tmp_2 = tmp_0.view(1, -1, 8, 32);  tmp_0 = None
        tmp_3 = tmp_2.transpose(1, 2);  tmp_2 = None
        tmp_4 = tmp_3.contiguous();  tmp_3 = None
        tmp_5 = in_3.view(1, -1, 8, 32);  in_3 = None
        tmp_6 = tmp_5.transpose(1, 2);  tmp_5 = None
        return (tmp_4, tmp_1, tmp_6)
        
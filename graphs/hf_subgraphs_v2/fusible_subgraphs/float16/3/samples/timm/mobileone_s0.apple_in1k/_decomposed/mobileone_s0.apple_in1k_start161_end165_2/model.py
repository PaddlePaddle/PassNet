import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        in_4 += in_5;  in_7 = in_4;  in_4 = in_5 = None
        in_7 += in_6;  tmp_4 = in_7;  in_7 = in_6 = None
        tmp_6 = torch.nn.functional.relu(tmp_4, inplace = True);  tmp_4 = None
        tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  in_0 = in_1 = in_3 = in_2 = None
        return (tmp_6, tmp_7)
        
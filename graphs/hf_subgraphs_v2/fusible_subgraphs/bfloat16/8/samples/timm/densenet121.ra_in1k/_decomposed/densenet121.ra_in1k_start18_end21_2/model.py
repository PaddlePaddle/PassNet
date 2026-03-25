import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_4 = torch.cat([in_4, in_5, in_6], 1);  in_4 = in_5 = in_6 = None
        tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_4 = in_0 = in_1 = in_3 = in_2 = None
        tmp_6 = torch.nn.functional.relu(tmp_5, inplace = True);  tmp_5 = None
        return (tmp_6,)
        
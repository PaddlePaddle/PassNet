import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        tmp_4 = in_5 * in_4;  in_5 = in_4 = None
        tmp_5 = tmp_4.sum(dim = 1);  tmp_4 = None
        tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_5 = in_0 = in_1 = in_3 = in_2 = None
        tmp_7 = torch.nn.functional.relu(tmp_6, inplace = True);  tmp_6 = None
        return (tmp_7,)
        
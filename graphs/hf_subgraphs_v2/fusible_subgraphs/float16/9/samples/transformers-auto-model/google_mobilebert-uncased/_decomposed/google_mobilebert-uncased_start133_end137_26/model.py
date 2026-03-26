import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = in_3 + in_2;  in_3 = in_2 = None
        tmp_3 = tmp_2 * in_1;  tmp_2 = in_1 = None
        tmp_4 = tmp_3 + in_0;  tmp_3 = in_0 = None
        tmp_5 = torch.tensor(1000);  tmp_5 = None
        return (tmp_4,)
        
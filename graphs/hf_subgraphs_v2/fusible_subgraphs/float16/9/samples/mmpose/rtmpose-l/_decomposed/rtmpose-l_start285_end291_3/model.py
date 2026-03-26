import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_1 = in_2 * in_1;  in_2 = in_1 = None
        tmp_2 = tmp_1 + in_0;  tmp_1 = in_0 = None
        unbind = torch.unbind(tmp_2, dim = 2);  tmp_2 = None
        tmp_4 = unbind[0]
        tmp_5 = unbind[1];  unbind = None
        tmp_6 = tmp_5.permute(0, 2, 1);  tmp_5 = None
        return (tmp_6, tmp_4)
        
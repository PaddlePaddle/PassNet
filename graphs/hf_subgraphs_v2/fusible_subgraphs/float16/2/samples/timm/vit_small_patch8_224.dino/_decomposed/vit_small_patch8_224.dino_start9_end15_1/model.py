import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = in_0.reshape(1, 785, 3, 6, 64);  in_0 = None
        tmp_1 = tmp_0.permute(2, 0, 3, 1, 4);  tmp_0 = None
        unbind = tmp_1.unbind(0);  tmp_1 = None
        tmp_3 = unbind[0]
        tmp_4 = unbind[1]
        tmp_5 = unbind[2];  unbind = None
        return (tmp_4, tmp_3, tmp_5)
        
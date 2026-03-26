import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = in_0 * in_3;  in_0 = in_3 = None
        tmp_3 = tmp_2 + in_2;  tmp_2 = in_2 = None
        tmp_4 = in_1[slice(None, 729, None)];  in_1 = None
        tmp_5 = tmp_4.reshape(1, 27, 27, -1);  tmp_4 = None
        tmp_6 = tmp_5.permute(0, 3, 1, 2);  tmp_5 = None
        return (tmp_3, tmp_6)
        
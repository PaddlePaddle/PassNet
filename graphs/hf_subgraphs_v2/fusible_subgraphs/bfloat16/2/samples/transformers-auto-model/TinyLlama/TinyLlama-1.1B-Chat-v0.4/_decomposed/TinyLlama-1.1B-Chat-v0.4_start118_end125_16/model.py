import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_0 = in_1 * in_0;  in_0 = None
        tmp_1 = in_1[(Ellipsis, slice(None, 32, None))]
        tmp_2 = in_1[(Ellipsis, slice(32, None, None))];  in_1 = None
        tmp_3 = -tmp_2;  tmp_2 = None
        tmp_4 = torch.cat((tmp_3, tmp_1), dim = -1);  tmp_3 = tmp_1 = None
        tmp_5 = tmp_4 * in_2;  tmp_4 = in_2 = None
        tmp_6 = tmp_0 + tmp_5;  tmp_0 = tmp_5 = None
        return (tmp_6,)
        
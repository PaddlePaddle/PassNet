import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_0 = in_1 * in_0;  in_0 = None
        tmp_1 = in_1[(Ellipsis, slice(None, 64, None))]
        tmp_2 = in_1[(Ellipsis, slice(64, None, None))];  in_1 = None
        tmp_3 = -tmp_2;  tmp_2 = None
        tmp_4 = torch.cat((tmp_3, tmp_1), dim = -1);  tmp_3 = tmp_1 = None
        tmp_5 = tmp_4 * in_2;  tmp_4 = in_2 = None
        tmp_6 = tmp_0 + tmp_5;  tmp_0 = tmp_5 = None
        tmp_7 = tmp_6[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None))];  tmp_6 = None
        tmp_8 = tmp_7.expand(32, 4, 4, 64, 128);  tmp_7 = None
        return (tmp_8,)
        
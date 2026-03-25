import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        tmp_0 = in_3 * in_1
        tmp_1 = in_3[(Ellipsis, slice(None, 64, None))]
        tmp_2 = in_3[(Ellipsis, slice(64, None, None))];  in_3 = None
        tmp_3 = -tmp_2;  tmp_2 = None
        tmp_4 = torch.cat((tmp_3, tmp_1), dim = -1);  tmp_3 = tmp_1 = None
        tmp_5 = tmp_4 * in_4;  tmp_4 = None
        tmp_6 = tmp_0 + tmp_5;  tmp_0 = tmp_5 = None
        tmp_7 = in_2 * in_1;  in_1 = None
        tmp_8 = in_2[(Ellipsis, slice(None, 64, None))]
        tmp_9 = in_2[(Ellipsis, slice(64, None, None))];  in_2 = None
        tmp_10 = -tmp_9;  tmp_9 = None
        tmp_11 = torch.cat((tmp_10, tmp_8), dim = -1);  tmp_10 = tmp_8 = None
        tmp_12 = tmp_11 * in_4;  tmp_11 = in_4 = None
        tmp_13 = tmp_7 + tmp_12;  tmp_7 = tmp_12 = None
        tmp_14 = in_0[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 3, None))];  in_0 = None
        tmp_15 = tmp_6.contiguous();  tmp_6 = None
        tmp_16 = tmp_13.contiguous()
        return (tmp_14, tmp_13, tmp_16, tmp_15)
        
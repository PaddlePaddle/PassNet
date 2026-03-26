import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = torch.cat([in_1, in_0]);  in_1 = in_0 = None
        tmp_1 = torch.arange(24)
        tmp_2 = torch.arange(24)
        meshgrid = torch.functional.meshgrid(tmp_1, tmp_2, indexing = 'ij');  tmp_1 = tmp_2 = None
        tmp_4 = meshgrid[0]
        tmp_5 = meshgrid[1];  meshgrid = None
        tmp_6 = torch.stack((tmp_4, tmp_5));  tmp_4 = tmp_5 = None
        tmp_7 = torch.flatten(tmp_6, 1);  tmp_6 = None
        tmp_8 = tmp_7[(slice(None, None, None), slice(None, None, None), None)]
        tmp_9 = tmp_7[(slice(None, None, None), None, slice(None, None, None))];  tmp_7 = None
        tmp_10 = tmp_8 - tmp_9;  tmp_8 = tmp_9 = None
        tmp_11 = tmp_10.permute(1, 2, 0);  tmp_10 = None
        tmp_12 = tmp_11.contiguous();  tmp_11 = None
        tmp_13 = tmp_12[(slice(None, None, None), slice(None, None, None), 0)]
        tmp_13 += 23;  tmp_14 = tmp_13;  tmp_13 = None
        tmp_12[(slice(None, None, None), slice(None, None, None), 0)] = tmp_14;  setitem = tmp_12;  tmp_14 = setitem = None
        tmp_16 = tmp_12[(slice(None, None, None), slice(None, None, None), 1)]
        tmp_16 += 23;  tmp_17 = tmp_16;  tmp_16 = None
        tmp_12[(slice(None, None, None), slice(None, None, None), 1)] = tmp_17;  setitem_1 = tmp_12;  tmp_17 = setitem_1 = None
        tmp_19 = tmp_12[(slice(None, None, None), slice(None, None, None), 0)]
        tmp_19 *= 47;  tmp_20 = tmp_19;  tmp_19 = None
        tmp_12[(slice(None, None, None), slice(None, None, None), 0)] = tmp_20;  setitem_2 = tmp_12;  tmp_20 = setitem_2 = None
        tmp_22 = torch.zeros(size = (577, 577), dtype = torch.int64)
        tmp_23 = tmp_12.sum(-1);  tmp_12 = None
        tmp_22[(slice(1, None, None), slice(1, None, None))] = tmp_23;  setitem_3 = tmp_22;  tmp_23 = setitem_3 = None
        tmp_22[(0, slice(0, None, None))] = 2209;  setitem_4 = tmp_22;  setitem_4 = None
        tmp_22[(slice(0, None, None), 0)] = 2210;  setitem_5 = tmp_22;  setitem_5 = None
        tmp_22[(0, 0)] = 2211;  setitem_6 = tmp_22;  setitem_6 = None
        tmp_28 = tmp_22.view(-1);  tmp_22 = None
        return (tmp_0, tmp_28)
        
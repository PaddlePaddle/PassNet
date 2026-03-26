import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_1 = in_3 * in_1;  in_1 = None
        tmp_2 = in_3[(Ellipsis, slice(1, None, 2))]
        tmp_3 = -tmp_2;  tmp_2 = None
        tmp_4 = in_3[(Ellipsis, slice(None, None, 2))];  in_3 = None
        tmp_5 = torch.stack([tmp_3, tmp_4], -1);  tmp_3 = tmp_4 = None
        tmp_6 = tmp_5.reshape((1, 12, 256, 64));  tmp_5 = None
        tmp_7 = tmp_6 * in_5;  tmp_6 = in_5 = None
        tmp_8 = tmp_1 + tmp_7;  tmp_1 = tmp_7 = None
        tmp_9 = torch.cat([in_2, tmp_8], dim = 2);  in_2 = tmp_8 = None
        tmp_10 = tmp_9.type_as(in_6);  tmp_9 = None
        tmp_11 = in_4[(slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None))]
        tmp_12 = in_4[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  in_4 = None
        tensor_split = in_0.tensor_split(2, -1);  in_0 = None
        tmp_14 = tensor_split[0]
        tmp_15 = tensor_split[1];  tensor_split = None
        tmp_16 = tmp_12 * tmp_15;  tmp_15 = None
        tmp_17 = tmp_12[(Ellipsis, slice(1, None, 2))]
        tmp_18 = -tmp_17;  tmp_17 = None
        tmp_19 = tmp_12[(Ellipsis, slice(None, None, 2))];  tmp_12 = None
        tmp_20 = torch.stack([tmp_18, tmp_19], -1);  tmp_18 = tmp_19 = None
        tmp_21 = tmp_20.reshape((1, 12, 256, 64));  tmp_20 = None
        tmp_22 = tmp_21 * tmp_14;  tmp_21 = tmp_14 = None
        tmp_23 = tmp_16 + tmp_22;  tmp_16 = tmp_22 = None
        tmp_24 = torch.cat([tmp_11, tmp_23], dim = 2);  tmp_11 = tmp_23 = None
        tmp_25 = tmp_24.type_as(in_6);  tmp_24 = in_6 = None
        return (tmp_25, tmp_10)
        
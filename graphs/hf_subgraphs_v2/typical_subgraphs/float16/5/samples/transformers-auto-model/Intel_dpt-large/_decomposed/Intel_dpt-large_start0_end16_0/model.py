import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        tmp_7 = in_4[(slice(None, None, None), slice(None, 1, None))]
        tmp_8 = in_4[(0, slice(1, None, None))];  in_4 = None
        tmp_9 = tmp_8.reshape(1, 24, 24, -1);  tmp_8 = None
        tmp_10 = tmp_9.permute(0, 3, 1, 2);  tmp_9 = None
        tmp_11 = torch.nn.functional.interpolate(tmp_10, size = (24, 24), mode = 'bilinear');  tmp_10 = None
        tmp_12 = tmp_11.permute(0, 2, 3, 1);  tmp_11 = None
        tmp_13 = tmp_12.reshape(1, 576, -1);  tmp_12 = None
        tmp_14 = torch.cat([tmp_7, tmp_13], dim = 1);  tmp_7 = tmp_13 = None
        conv2d = torch.conv2d(in_0, in_2, in_1, (16, 16), (0, 0), (1, 1), 1);  in_0 = in_2 = in_1 = None
        tmp_16 = conv2d.flatten(2);  conv2d = None
        tmp_17 = tmp_16.transpose(1, 2);  tmp_16 = None
        tmp_18 = in_3.expand(1, -1, -1);  in_3 = None
        tmp_19 = torch.cat((tmp_18, tmp_17), dim = 1);  tmp_18 = tmp_17 = None
        tmp_20 = tmp_19 + tmp_14;  tmp_19 = tmp_14 = None
        tmp_21 = torch.nn.functional.dropout(tmp_20, 0.0, False, False);  tmp_20 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (1024,), in_6, in_5, 1e-12);  in_6 = in_5 = None
        return (tmp_21, tmp_22)
        
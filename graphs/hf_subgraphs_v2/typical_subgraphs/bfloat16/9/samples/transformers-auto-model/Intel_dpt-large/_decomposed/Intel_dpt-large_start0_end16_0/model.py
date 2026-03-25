import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor):
        tmp_7 = w_3[(slice(None, None, None), slice(None, 1, None))]
        tmp_8 = w_3[(0, slice(1, None, None))];  w_3 = None
        tmp_9 = tmp_8.reshape(1, 24, 24, -1);  tmp_8 = None
        tmp_10 = tmp_9.permute(0, 3, 1, 2);  tmp_9 = None
        tmp_11 = torch.nn.functional.interpolate(tmp_10, size = (24, 24), mode = 'bilinear');  tmp_10 = None
        tmp_12 = tmp_11.permute(0, 2, 3, 1);  tmp_11 = None
        tmp_13 = tmp_12.reshape(1, 576, -1);  tmp_12 = None
        tmp_14 = torch.cat([tmp_7, tmp_13], dim = 1);  tmp_7 = tmp_13 = None
        conv2d = torch.conv2d(in_0, w_1, w_0, (16, 16), (0, 0), (1, 1), 1);  in_0 = w_1 = w_0 = None
        tmp_16 = conv2d.flatten(2);  conv2d = None
        tmp_17 = tmp_16.transpose(1, 2);  tmp_16 = None
        tmp_18 = w_2.expand(1, -1, -1);  w_2 = None
        tmp_19 = torch.cat((tmp_18, tmp_17), dim = 1);  tmp_18 = tmp_17 = None
        tmp_20 = tmp_19 + tmp_14;  tmp_19 = tmp_14 = None
        tmp_21 = torch.nn.functional.dropout(tmp_20, 0.0, False, False);  tmp_20 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (1024,), w_5, w_4, 1e-12);  w_5 = w_4 = None
        return (tmp_21, tmp_22)
        
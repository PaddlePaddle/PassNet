import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_1, w_0, (16, 16), (0, 0), (1, 1), 1);  in_0 = w_1 = w_0 = None
        tmp_9 = conv2d.flatten(2);  conv2d = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        tmp_11 = w_2.expand(1, -1, -1);  w_2 = None
        tmp_12 = w_3.expand(1, -1, -1);  w_3 = None
        tmp_13 = torch.cat((tmp_11, tmp_10, tmp_12), dim = 1);  tmp_11 = tmp_10 = tmp_12 = None
        tmp_14 = w_4[(slice(None, None, None), 0, slice(None, None, None))]
        tmp_15 = tmp_14[(slice(None, None, None), None)];  tmp_14 = None
        tmp_16 = w_4[(slice(None, None, None), slice(-100, None, None), slice(None, None, None))]
        tmp_17 = w_4[(slice(None, None, None), slice(1, -100, None), slice(None, None, None))];  w_4 = None
        tmp_18 = tmp_17.transpose(1, 2);  tmp_17 = None
        tmp_19 = tmp_18.view(1, 192, 50, 83);  tmp_18 = None
        tmp_20 = torch.nn.functional.interpolate(tmp_19, size = (32, 32), mode = 'bicubic', align_corners = False);  tmp_19 = None
        tmp_21 = tmp_20.flatten(2);  tmp_20 = None
        tmp_22 = tmp_21.transpose(1, 2);  tmp_21 = None
        tmp_23 = torch.cat((tmp_15, tmp_22, tmp_16), dim = 1);  tmp_15 = tmp_22 = tmp_16 = None
        tmp_24 = tmp_13 + tmp_23;  tmp_13 = tmp_23 = None
        tmp_25 = torch.nn.functional.dropout(tmp_24, 0.0, False, False);  tmp_24 = None
        tmp_26 = torch.nn.functional.layer_norm(tmp_25, (192,), w_6, w_5, 1e-12);  w_6 = w_5 = None
        return (tmp_25, tmp_26)
        
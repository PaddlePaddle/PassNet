import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor):
        tmp_13 = torch.nn.functional.interpolate(in_12, size = (240, 240), mode = 'bicubic', align_corners = False)
        conv2d = torch.conv2d(tmp_13, in_5, in_4, (12, 12), (0, 0), (1, 1), 1);  tmp_13 = in_5 = in_4 = None
        tmp_15 = conv2d.flatten(2);  conv2d = None
        tmp_16 = tmp_15.transpose(1, 2);  tmp_15 = None
        tmp_17 = in_8.expand(1, -1, -1);  in_8 = None
        tmp_18 = torch.cat((tmp_17, tmp_16), dim = 1);  tmp_17 = tmp_16 = None
        tmp_19 = tmp_18 + in_10;  tmp_18 = in_10 = None
        tmp_20 = torch.nn.functional.dropout(tmp_19, 0.0, False, False);  tmp_19 = None
        conv2d_1 = torch.conv2d(in_12, in_7, in_6, (16, 16), (0, 0), (1, 1), 1);  in_12 = in_7 = in_6 = None
        tmp_22 = conv2d_1.flatten(2);  conv2d_1 = None
        tmp_23 = tmp_22.transpose(1, 2);  tmp_22 = None
        tmp_24 = in_9.expand(1, -1, -1);  in_9 = None
        tmp_25 = torch.cat((tmp_24, tmp_23), dim = 1);  tmp_24 = tmp_23 = None
        tmp_26 = tmp_25 + in_11;  tmp_25 = in_11 = None
        tmp_27 = torch.nn.functional.dropout(tmp_26, 0.0, False, False);  tmp_26 = None
        tmp_28 = torch.nn.functional.layer_norm(tmp_20, (192,), in_3, in_2, 1e-06);  in_3 = in_2 = None
        linear = torch.nn.functional.linear(tmp_28, in_1, in_0);  tmp_28 = in_1 = in_0 = None
        return (linear, tmp_20, tmp_27)
        
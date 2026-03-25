import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_1, w_0, (2, 2), (0, 0), (1, 1), 1);  in_0 = w_1 = w_0 = None
        tmp_10 = conv2d.flatten(2);  conv2d = None
        tmp_11 = tmp_10.transpose(1, 2);  tmp_10 = None
        tmp_12 = w_2.expand(1, -1, -1);  w_2 = None
        tmp_13 = w_3.expand(1, -1, -1);  w_3 = None
        tmp_14 = torch.cat((tmp_12, tmp_11, tmp_13), dim = 1);  tmp_12 = tmp_11 = tmp_13 = None
        tmp_15 = w_4[(slice(None, None, None), 0, slice(None, None, None))]
        tmp_16 = tmp_15[(slice(None, None, None), None)];  tmp_15 = None
        tmp_17 = w_4[(slice(None, None, None), slice(-10, None, None), slice(None, None, None))]
        tmp_18 = w_4[(slice(None, None, None), slice(1, -10, None), slice(None, None, None))];  w_4 = None
        tmp_19 = tmp_18.transpose(1, 2);  tmp_18 = None
        tmp_20 = tmp_19.view(1, 32, 15, 15);  tmp_19 = None
        tmp_21 = torch.nn.functional.interpolate(tmp_20, size = (15, 15), mode = 'bicubic', align_corners = False);  tmp_20 = None
        tmp_22 = tmp_21.flatten(2);  tmp_21 = None
        tmp_23 = tmp_22.transpose(1, 2);  tmp_22 = None
        tmp_24 = torch.cat((tmp_16, tmp_23, tmp_17), dim = 1);  tmp_16 = tmp_23 = tmp_17 = None
        tmp_25 = tmp_14 + tmp_24;  tmp_14 = tmp_24 = None
        tmp_26 = torch.nn.functional.dropout(tmp_25, 0.1, False, False);  tmp_25 = None
        tmp_27 = w_7[(slice(None, None, None), slice(None, None, None), 0, slice(None, None, None))]
        tmp_28 = tmp_27[(slice(None, None, None), None)];  tmp_27 = None
        tmp_29 = w_7[(slice(None, None, None), slice(None, None, None), slice(-10, None, None), slice(None, None, None))]
        tmp_30 = w_7[(slice(None, None, None), slice(None, None, None), slice(1, -10, None), slice(None, None, None))];  w_7 = None
        tmp_31 = tmp_30.transpose(2, 3);  tmp_30 = None
        tmp_32 = tmp_31.view(4, 32, 15, 15);  tmp_31 = None
        tmp_33 = torch.nn.functional.interpolate(tmp_32, size = (15, 15), mode = 'bicubic', align_corners = False);  tmp_32 = None
        tmp_34 = tmp_33.flatten(2);  tmp_33 = None
        tmp_35 = tmp_34.transpose(1, 2);  tmp_34 = None
        tmp_36 = tmp_35.contiguous();  tmp_35 = None
        tmp_37 = tmp_36.view(4, 1, 225, 32);  tmp_36 = None
        tmp_38 = torch.cat((tmp_28, tmp_37, tmp_29), dim = 2);  tmp_28 = tmp_37 = tmp_29 = None
        tmp_39 = torch.nn.functional.layer_norm(tmp_26, (32,), w_6, w_5, 1e-12);  w_6 = w_5 = None
        return (tmp_26, tmp_39, tmp_38)
        
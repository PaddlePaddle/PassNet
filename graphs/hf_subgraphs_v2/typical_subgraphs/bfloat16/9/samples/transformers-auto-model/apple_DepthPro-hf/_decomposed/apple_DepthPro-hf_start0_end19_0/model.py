import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor):
        tmp_7 = torch.nn.functional.interpolate(in_0, scale_factor = 0.25, mode = 'bilinear', align_corners = False)
        tmp_8 = torch.nn.functional.interpolate(in_0, scale_factor = 0.5, mode = 'bilinear', align_corners = False)
        tmp_9 = torch.nn.functional.interpolate(in_0, scale_factor = 1, mode = 'bilinear', align_corners = False);  in_0 = None
        tmp_10 = torch.nn.functional.unfold(tmp_8, kernel_size = (384, 384), stride = (192, 192));  tmp_8 = None
        tmp_11 = tmp_10.permute(2, 0, 1);  tmp_10 = None
        tmp_12 = tmp_11.reshape(-1, 3, 384, 384);  tmp_11 = None
        tmp_13 = torch.nn.functional.unfold(tmp_9, kernel_size = (384, 384), stride = (288, 288));  tmp_9 = None
        tmp_14 = tmp_13.permute(2, 0, 1);  tmp_13 = None
        tmp_15 = tmp_14.reshape(-1, 3, 384, 384);  tmp_14 = None
        tmp_16 = torch.cat([tmp_15, tmp_12, tmp_7], dim = 0);  tmp_15 = tmp_12 = tmp_7 = None
        tmp_17 = tmp_16.to(dtype = torch.float16);  tmp_16 = None
        conv2d = torch.conv2d(tmp_17, w_1, w_0, (16, 16), (0, 0), (1, 1), 1);  tmp_17 = w_1 = w_0 = None
        tmp_19 = conv2d.flatten(2);  conv2d = None
        tmp_20 = tmp_19.transpose(1, 2);  tmp_19 = None
        tmp_21 = w_2.expand(35, -1, -1);  w_2 = None
        tmp_22 = torch.cat((tmp_21, tmp_20), dim = 1);  tmp_21 = tmp_20 = None
        tmp_23 = tmp_22 + w_3;  tmp_22 = w_3 = None
        tmp_24 = torch.nn.functional.dropout(tmp_23, 0.0, False, False);  tmp_23 = None
        tmp_25 = torch.nn.functional.layer_norm(tmp_24, (1024,), w_5, w_4, 1e-06);  w_5 = w_4 = None
        return (tmp_24, tmp_25)
        
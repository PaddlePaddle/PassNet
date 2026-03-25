import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1, in_2):
        tmp_10 = in_2.transpose(1, 2);  in_2 = None
        tmp_11 = tmp_10.view(1, 2048, 12, 12);  tmp_10 = None
        conv2d = torch.conv2d(tmp_11, w_7, w_6, (1, 1), (1, 1), (1, 1), 2048);  tmp_11 = w_7 = w_6 = None
        tmp_13 = conv2d.flatten(2);  conv2d = None
        tmp_14 = tmp_13.transpose(1, 2);  tmp_13 = None
        tmp_15 = torch.nn.functional.gelu(tmp_14);  tmp_14 = None
        tmp_16 = torch.nn.functional.dropout(tmp_15, 0.0, False, False);  tmp_15 = None
        linear = torch.nn.functional.linear(tmp_16, w_5, w_4);  tmp_16 = w_5 = w_4 = None
        tmp_18 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_19 = tmp_18 + in_1;  tmp_18 = in_1 = None
        tmp_20 = torch.nn.functional.layer_norm(tmp_19, (512,), w_9, w_8, 1e-05);  tmp_19 = w_9 = w_8 = None
        tmp_21 = tmp_20.reshape(1, 12, 12, -1);  tmp_20 = None
        tmp_22 = tmp_21.permute(0, 3, 1, 2);  tmp_21 = None
        tmp_23 = tmp_22.contiguous();  tmp_22 = None
        conv2d_1 = torch.conv2d(tmp_23, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_23 = w_1 = w_0 = None
        tmp_25 = torch.nn.functional.interpolate(conv2d_1, None, 2.0, 'bilinear', False, recompute_scale_factor = None);  conv2d_1 = None
        conv2d_2 = torch.conv2d(in_0, w_3, w_2, (1, 1), (0, 0), (1, 1), 1);  in_0 = w_3 = w_2 = None
        tmp_27 = torch.cat((conv2d_2, tmp_25), dim = 1)
        return (tmp_27, tmp_25, conv2d_2)
        
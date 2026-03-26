import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13):
        tmp_12 = in_13.transpose(1, 2);  in_13 = None
        tmp_13 = tmp_12.view(32, 1280, 32, 32);  tmp_12 = None
        conv2d = torch.conv2d(tmp_13, in_3, in_2, (1, 1), (1, 1), (1, 1), 1280);  tmp_13 = in_3 = in_2 = None
        tmp_15 = conv2d.flatten(2);  conv2d = None
        tmp_16 = tmp_15.transpose(1, 2);  tmp_15 = None
        tmp_17 = torch.nn.functional.gelu(tmp_16);  tmp_16 = None
        tmp_18 = torch.nn.functional.dropout(tmp_17, 0.0, False, False);  tmp_17 = None
        linear = torch.nn.functional.linear(tmp_18, in_1, in_0);  tmp_18 = in_1 = in_0 = None
        tmp_20 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_21 = tmp_20 + in_12;  tmp_20 = in_12 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (320,), in_7, in_6, 1e-05);  tmp_21 = in_7 = in_6 = None
        tmp_23 = tmp_22.reshape(32, 32, 32, -1);  tmp_22 = None
        tmp_24 = tmp_23.permute(0, 3, 1, 2);  tmp_23 = None
        tmp_25 = tmp_24.contiguous();  tmp_24 = None
        conv2d_1 = torch.conv2d(tmp_25, in_11, in_10, (2, 2), (1, 1), (1, 1), 1);  in_11 = in_10 = None
        tmp_27 = conv2d_1.flatten(2);  conv2d_1 = None
        tmp_28 = tmp_27.transpose(1, 2);  tmp_27 = None
        tmp_29 = torch.nn.functional.layer_norm(tmp_28, (512,), in_9, in_8, 1e-05);  tmp_28 = in_9 = in_8 = None
        tmp_30 = torch.nn.functional.layer_norm(tmp_29, (512,), in_5, in_4, 1e-05);  in_5 = in_4 = None
        return (tmp_29, tmp_25, tmp_30)
        
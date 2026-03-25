import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0, in_1):
        tmp_12 = in_1.transpose(1, 2);  in_1 = None
        tmp_13 = tmp_12.view(1, 256, 4, 4);  tmp_12 = None
        conv2d = torch.conv2d(tmp_13, w_3, w_2, (1, 1), (1, 1), (1, 1), 256);  tmp_13 = w_3 = w_2 = None
        tmp_15 = conv2d.flatten(2);  conv2d = None
        tmp_16 = tmp_15.transpose(1, 2);  tmp_15 = None
        tmp_17 = torch.nn.functional.gelu(tmp_16);  tmp_16 = None
        tmp_18 = torch.nn.functional.dropout(tmp_17, 0.1, False, False);  tmp_17 = None
        linear = torch.nn.functional.linear(tmp_18, w_1, w_0);  tmp_18 = w_1 = w_0 = None
        tmp_20 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_21 = tmp_20 + in_0;  tmp_20 = in_0 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (64,), w_7, w_6, 1e-05);  tmp_21 = w_7 = w_6 = None
        tmp_23 = tmp_22.reshape(1, 4, 4, -1);  tmp_22 = None
        tmp_24 = tmp_23.permute(0, 3, 1, 2);  tmp_23 = None
        tmp_25 = tmp_24.contiguous();  tmp_24 = None
        conv2d_1 = torch.conv2d(tmp_25, w_11, w_10, (2, 2), (1, 1), (1, 1), 1);  tmp_25 = w_11 = w_10 = None
        tmp_27 = conv2d_1.flatten(2);  conv2d_1 = None
        tmp_28 = tmp_27.transpose(1, 2);  tmp_27 = None
        tmp_29 = torch.nn.functional.layer_norm(tmp_28, (128,), w_9, w_8, 1e-05);  tmp_28 = w_9 = w_8 = None
        tmp_30 = torch.nn.functional.layer_norm(tmp_29, (128,), w_5, w_4, 1e-05);  w_5 = w_4 = None
        return (tmp_29, tmp_30)
        
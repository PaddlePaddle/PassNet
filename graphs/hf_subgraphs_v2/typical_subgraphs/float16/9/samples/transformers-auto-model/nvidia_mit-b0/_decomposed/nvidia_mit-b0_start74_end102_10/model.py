import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, in_0, in_1):
        tmp_18 = in_1.transpose(1, 2);  in_1 = None
        tmp_19 = tmp_18.view(1, 128, 128, 128);  tmp_18 = None
        conv2d = torch.conv2d(tmp_19, w_3, w_2, (1, 1), (1, 1), (1, 1), 128);  tmp_19 = w_3 = w_2 = None
        tmp_21 = conv2d.flatten(2);  conv2d = None
        tmp_22 = tmp_21.transpose(1, 2);  tmp_21 = None
        tmp_23 = torch.nn.functional.gelu(tmp_22);  tmp_22 = None
        tmp_24 = torch.nn.functional.dropout(tmp_23, 0.0, False, False);  tmp_23 = None
        linear = torch.nn.functional.linear(tmp_24, w_1, w_0);  tmp_24 = w_1 = w_0 = None
        tmp_26 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_27 = tmp_26 + in_0;  tmp_26 = in_0 = None
        tmp_28 = torch.nn.functional.layer_norm(tmp_27, (32,), w_13, w_12, 1e-05);  tmp_27 = w_13 = w_12 = None
        tmp_29 = tmp_28.reshape(1, 128, 128, -1);  tmp_28 = None
        tmp_30 = tmp_29.permute(0, 3, 1, 2);  tmp_29 = None
        tmp_31 = tmp_30.contiguous();  tmp_30 = None
        conv2d_1 = torch.conv2d(tmp_31, w_17, w_16, (2, 2), (1, 1), (1, 1), 1);  tmp_31 = w_17 = w_16 = None
        tmp_33 = conv2d_1.flatten(2);  conv2d_1 = None
        tmp_34 = tmp_33.transpose(1, 2);  tmp_33 = None
        tmp_35 = torch.nn.functional.layer_norm(tmp_34, (64,), w_15, w_14, 1e-05);  tmp_34 = w_15 = w_14 = None
        tmp_36 = torch.nn.functional.layer_norm(tmp_35, (64,), w_11, w_10, 1e-05);  w_11 = w_10 = None
        linear_1 = torch.nn.functional.linear(tmp_36, w_7, w_6);  w_7 = w_6 = None
        tmp_38 = linear_1.view(1, -1, 2, 32);  linear_1 = None
        tmp_39 = tmp_38.transpose(1, 2);  tmp_38 = None
        tmp_40 = tmp_36.permute(0, 2, 1);  tmp_36 = None
        tmp_41 = tmp_40.reshape(1, 64, 64, 64);  tmp_40 = None
        conv2d_2 = torch.conv2d(tmp_41, w_9, w_8, (4, 4), (0, 0), (1, 1), 1);  tmp_41 = w_9 = w_8 = None
        tmp_43 = conv2d_2.reshape(1, 64, -1);  conv2d_2 = None
        tmp_44 = tmp_43.permute(0, 2, 1);  tmp_43 = None
        tmp_45 = torch.nn.functional.layer_norm(tmp_44, (64,), w_5, w_4, 1e-05);  tmp_44 = w_5 = w_4 = None
        return (tmp_35, tmp_45, tmp_39)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19):
        tmp_18 = in_19.transpose(1, 2);  in_19 = None
        tmp_19 = tmp_18.view(16, 128, 128, 128);  tmp_18 = None
        to = tmp_19.to(torch.float16);  tmp_19 = None
        conv2d = torch.conv2d(to, in_3, in_2, (1, 1), (1, 1), (1, 1), 128);  to = in_3 = in_2 = None
        tmp_21 = conv2d.flatten(2);  conv2d = None
        tmp_22 = tmp_21.transpose(1, 2);  tmp_21 = None
        tmp_23 = torch.nn.functional.gelu(tmp_22);  tmp_22 = None
        tmp_24 = torch.nn.functional.dropout(tmp_23, 0.0, False, False);  tmp_23 = None
        to_1 = tmp_24.to(torch.float16);  tmp_24 = None
        linear = torch.nn.functional.linear(to_1, in_1, in_0);  to_1 = in_1 = in_0 = None
        tmp_26 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_27 = tmp_26 + in_18;  tmp_26 = in_18 = None
        tmp_28 = torch.nn.functional.layer_norm(tmp_27, (32,), in_13, in_12, 1e-05);  tmp_27 = in_13 = in_12 = None
        tmp_29 = tmp_28.reshape(16, 128, 128, -1);  tmp_28 = None
        tmp_30 = tmp_29.permute(0, 3, 1, 2);  tmp_29 = None
        tmp_31 = tmp_30.contiguous();  tmp_30 = None
        to_2 = tmp_31.to(torch.float16)
        conv2d_1 = torch.conv2d(to_2, in_17, in_16, (2, 2), (1, 1), (1, 1), 1);  to_2 = in_17 = in_16 = None
        tmp_33 = conv2d_1.flatten(2);  conv2d_1 = None
        tmp_34 = tmp_33.transpose(1, 2);  tmp_33 = None
        tmp_35 = torch.nn.functional.layer_norm(tmp_34, (64,), in_15, in_14, 1e-05);  tmp_34 = in_15 = in_14 = None
        tmp_36 = torch.nn.functional.layer_norm(tmp_35, (64,), in_11, in_10, 1e-05);  in_11 = in_10 = None
        to_3 = tmp_36.to(torch.float16)
        linear_1 = torch.nn.functional.linear(to_3, in_7, in_6);  to_3 = in_7 = in_6 = None
        tmp_38 = linear_1.view(16, -1, 2, 32);  linear_1 = None
        tmp_39 = tmp_38.transpose(1, 2);  tmp_38 = None
        tmp_40 = tmp_36.permute(0, 2, 1);  tmp_36 = None
        tmp_41 = tmp_40.reshape(16, 64, 64, 64);  tmp_40 = None
        to_4 = tmp_41.to(torch.float16);  tmp_41 = None
        conv2d_2 = torch.conv2d(to_4, in_9, in_8, (4, 4), (0, 0), (1, 1), 1);  to_4 = in_9 = in_8 = None
        tmp_43 = conv2d_2.reshape(16, 64, -1);  conv2d_2 = None
        tmp_44 = tmp_43.permute(0, 2, 1);  tmp_43 = None
        tmp_45 = torch.nn.functional.layer_norm(tmp_44, (64,), in_5, in_4, 1e-05);  tmp_44 = in_5 = in_4 = None
        return (tmp_35, tmp_31, tmp_45, tmp_39)
        
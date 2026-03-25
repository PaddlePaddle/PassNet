import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, in_0, in_1, in_2, in_3, in_4):
        tmp_14 = in_4.transpose(1, 2);  in_4 = None
        tmp_15 = tmp_14.view(1, 2048, 16, 16);  tmp_14 = None
        conv2d = torch.conv2d(tmp_15, w_11, w_10, (1, 1), (1, 1), (1, 1), 2048);  tmp_15 = w_11 = w_10 = None
        tmp_17 = conv2d.flatten(2);  conv2d = None
        tmp_18 = tmp_17.transpose(1, 2);  tmp_17 = None
        tmp_19 = torch.nn.functional.gelu(tmp_18);  tmp_18 = None
        tmp_20 = torch.nn.functional.dropout(tmp_19, 0.0, False, False);  tmp_19 = None
        linear = torch.nn.functional.linear(tmp_20, w_9, w_8);  tmp_20 = w_9 = w_8 = None
        tmp_22 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_23 = tmp_22 + in_3;  tmp_22 = in_3 = None
        tmp_24 = torch.nn.functional.layer_norm(tmp_23, (512,), w_13, w_12, 1e-05);  tmp_23 = w_13 = w_12 = None
        tmp_25 = tmp_24.reshape(1, 16, 16, -1);  tmp_24 = None
        tmp_26 = tmp_25.permute(0, 3, 1, 2);  tmp_25 = None
        tmp_27 = tmp_26.contiguous();  tmp_26 = None
        tmp_28 = in_1.flatten(2);  in_1 = None
        tmp_29 = tmp_28.transpose(1, 2);  tmp_28 = None
        linear_1 = torch.nn.functional.linear(tmp_29, w_1, w_0);  tmp_29 = w_1 = w_0 = None
        tmp_31 = linear_1.permute(0, 2, 1);  linear_1 = None
        tmp_32 = tmp_31.reshape(1, -1, 128, 128);  tmp_31 = None
        tmp_33 = torch.nn.functional.interpolate(tmp_32, size = (128, 128), mode = 'bilinear', align_corners = False);  tmp_32 = None
        tmp_34 = in_0.flatten(2);  in_0 = None
        tmp_35 = tmp_34.transpose(1, 2);  tmp_34 = None
        linear_2 = torch.nn.functional.linear(tmp_35, w_3, w_2);  tmp_35 = w_3 = w_2 = None
        tmp_37 = linear_2.permute(0, 2, 1);  linear_2 = None
        tmp_38 = tmp_37.reshape(1, -1, 64, 64);  tmp_37 = None
        tmp_39 = torch.nn.functional.interpolate(tmp_38, size = (128, 128), mode = 'bilinear', align_corners = False);  tmp_38 = None
        tmp_40 = in_2.flatten(2);  in_2 = None
        tmp_41 = tmp_40.transpose(1, 2);  tmp_40 = None
        linear_3 = torch.nn.functional.linear(tmp_41, w_5, w_4);  tmp_41 = w_5 = w_4 = None
        tmp_43 = linear_3.permute(0, 2, 1);  linear_3 = None
        tmp_44 = tmp_43.reshape(1, -1, 32, 32);  tmp_43 = None
        tmp_45 = torch.nn.functional.interpolate(tmp_44, size = (128, 128), mode = 'bilinear', align_corners = False);  tmp_44 = None
        tmp_46 = tmp_27.flatten(2);  tmp_27 = None
        tmp_47 = tmp_46.transpose(1, 2);  tmp_46 = None
        linear_4 = torch.nn.functional.linear(tmp_47, w_7, w_6);  tmp_47 = w_7 = w_6 = None
        tmp_49 = linear_4.permute(0, 2, 1);  linear_4 = None
        tmp_50 = tmp_49.reshape(1, -1, 16, 16);  tmp_49 = None
        tmp_51 = torch.nn.functional.interpolate(tmp_50, size = (128, 128), mode = 'bilinear', align_corners = False);  tmp_50 = None
        tmp_52 = torch.cat((tmp_51, tmp_45, tmp_39, tmp_33), dim = 1);  tmp_51 = tmp_45 = tmp_39 = tmp_33 = None
        return (tmp_52,)
        
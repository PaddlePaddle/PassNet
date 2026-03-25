import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20):
        tmp_19 = torch.nn.functional.gelu(in_20, approximate = 'none');  in_20 = None
        tmp_20 = torch.nn.functional.dropout(tmp_19, 0.0, False, False);  tmp_19 = None
        linear = torch.nn.functional.linear(tmp_20, in_5, in_4);  tmp_20 = in_5 = in_4 = None
        tmp_22 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_23 = in_19 + tmp_22;  in_19 = tmp_22 = None
        tmp_24 = tmp_23[(slice(None, None, None), slice(1, None, None), slice(None, None, None))];  tmp_23 = None
        tmp_25 = tmp_24.reshape(1, 56, 56, -1);  tmp_24 = None
        tmp_26 = tmp_25.permute(0, 3, 1, 2);  tmp_25 = None
        tmp_27 = tmp_26.contiguous();  tmp_26 = None
        conv2d = torch.conv2d(tmp_27, in_3, in_2, (2, 2), (0, 0), (1, 1), 1);  tmp_27 = in_3 = in_2 = None
        tmp_29 = conv2d.flatten(2);  conv2d = None
        tmp_30 = tmp_29.transpose(1, 2);  tmp_29 = None
        tmp_31 = torch.nn.functional.layer_norm(tmp_30, (256,), in_1, in_0, 1e-05);  tmp_30 = in_1 = in_0 = None
        tmp_32 = in_18.expand(1, -1, -1);  in_18 = None
        tmp_33 = torch.cat((tmp_32, tmp_31), dim = 1);  tmp_32 = tmp_31 = None
        tmp_34 = tmp_33[(slice(None, None, None), slice(None, 1, None))]
        tmp_35 = tmp_33[(slice(None, None, None), slice(1, None, None))];  tmp_33 = None
        tmp_36 = tmp_35.transpose(1, 2);  tmp_35 = None
        tmp_37 = tmp_36.view(1, 256, 28, 28);  tmp_36 = None
        conv2d_1 = torch.conv2d(tmp_37, in_7, in_6, (1, 1), (1, 1), (1, 1), 256);  in_7 = in_6 = None
        tmp_39 = conv2d_1 + tmp_37;  conv2d_1 = tmp_37 = None
        tmp_40 = tmp_39.flatten(2);  tmp_39 = None
        tmp_41 = tmp_40.transpose(1, 2);  tmp_40 = None
        tmp_42 = torch.cat((tmp_34, tmp_41), dim = 1);  tmp_34 = tmp_41 = None
        tmp_43 = torch.nn.functional.layer_norm(tmp_42, (256,), in_17, in_16, 1e-06);  in_17 = in_16 = None
        linear_1 = torch.nn.functional.linear(tmp_43, in_15, in_14);  tmp_43 = in_15 = in_14 = None
        tmp_45 = linear_1.reshape(1, 785, 3, 8, 32);  linear_1 = None
        tmp_46 = tmp_45.permute(2, 0, 3, 1, 4);  tmp_45 = None
        unbind = tmp_46.unbind(0);  tmp_46 = None
        tmp_48 = unbind[0]
        tmp_49 = unbind[1]
        tmp_50 = unbind[2];  unbind = None
        tmp_51 = tmp_49.softmax(dim = 2);  tmp_49 = None
        tmp_52 = tmp_51.transpose(-1, -2);  tmp_51 = None
        matmul = tmp_52 @ tmp_50;  tmp_52 = None
        matmul_1 = tmp_48 @ matmul;  matmul = None
        tmp_55 = tmp_48[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  tmp_48 = None
        tmp_56 = tmp_50[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  tmp_50 = None
        tmp_57 = tmp_56.transpose(-1, -2);  tmp_56 = None
        tmp_58 = tmp_57.reshape(1, 256, 28, 28);  tmp_57 = None
        split = torch.functional.split(tmp_58, [64, 96, 96], dim = 1);  tmp_58 = None
        tmp_60 = split[0]
        tmp_61 = split[1]
        tmp_62 = split[2];  split = None
        conv2d_2 = torch.conv2d(tmp_60, in_9, in_8, (1, 1), (1, 1), (1, 1), 64);  tmp_60 = in_9 = in_8 = None
        conv2d_3 = torch.conv2d(tmp_61, in_11, in_10, (1, 1), (2, 2), (1, 1), 96);  tmp_61 = in_11 = in_10 = None
        conv2d_4 = torch.conv2d(tmp_62, in_13, in_12, (1, 1), (3, 3), (1, 1), 96);  tmp_62 = in_13 = in_12 = None
        tmp_66 = torch.cat([conv2d_2, conv2d_3, conv2d_4], dim = 1);  conv2d_2 = conv2d_3 = conv2d_4 = None
        tmp_67 = tmp_66.reshape(1, 8, 32, 784);  tmp_66 = None
        tmp_68 = tmp_67.transpose(-1, -2);  tmp_67 = None
        tmp_69 = tmp_55 * tmp_68;  tmp_55 = tmp_68 = None
        tmp_70 = torch.nn.functional.pad(tmp_69, (0, 0, 1, 0, 0, 0), 'constant', None);  tmp_69 = None
        tmp_71 = 0.1767766952966369 * matmul_1;  matmul_1 = None
        tmp_72 = tmp_71 + tmp_70;  tmp_71 = tmp_70 = None
        tmp_73 = tmp_72.transpose(1, 2);  tmp_72 = None
        tmp_74 = tmp_73.reshape(1, 785, 256);  tmp_73 = None
        return (tmp_42, tmp_74)
        
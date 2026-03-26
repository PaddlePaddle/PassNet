import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, in_0, in_1):
        tmp_19 = torch.nn.functional.gelu(in_1, approximate = 'none');  in_1 = None
        tmp_20 = torch.nn.functional.dropout(tmp_19, 0.0, False, False);  tmp_19 = None
        linear = torch.nn.functional.linear(tmp_20, w_5, w_4);  tmp_20 = w_5 = w_4 = None
        tmp_22 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_23 = in_0 + tmp_22;  in_0 = tmp_22 = None
        tmp_24 = tmp_23[(slice(None, None, None), slice(1, None, None), slice(None, None, None))]
        tmp_25 = tmp_24.reshape(1, 28, 28, -1);  tmp_24 = None
        tmp_26 = tmp_25.permute(0, 3, 1, 2);  tmp_25 = None
        tmp_27 = tmp_26.contiguous();  tmp_26 = None
        conv2d = torch.conv2d(tmp_27, w_3, w_2, (2, 2), (0, 0), (1, 1), 1);  tmp_27 = w_3 = w_2 = None
        tmp_29 = conv2d.flatten(2);  conv2d = None
        tmp_30 = tmp_29.transpose(1, 2);  tmp_29 = None
        tmp_31 = torch.nn.functional.layer_norm(tmp_30, (216,), w_1, w_0, 1e-05);  tmp_30 = w_1 = w_0 = None
        tmp_32 = w_18.expand(1, -1, -1);  w_18 = None
        tmp_33 = torch.cat((tmp_32, tmp_31), dim = 1);  tmp_32 = tmp_31 = None
        tmp_34 = tmp_33[(slice(None, None, None), slice(None, 1, None))]
        tmp_35 = tmp_33[(slice(None, None, None), slice(1, None, None))];  tmp_33 = None
        tmp_36 = tmp_35.transpose(1, 2);  tmp_35 = None
        tmp_37 = tmp_36.view(1, 216, 14, 14);  tmp_36 = None
        conv2d_1 = torch.conv2d(tmp_37, w_7, w_6, (1, 1), (1, 1), (1, 1), 216);  w_7 = w_6 = None
        tmp_39 = conv2d_1 + tmp_37;  conv2d_1 = tmp_37 = None
        tmp_40 = tmp_39.flatten(2);  tmp_39 = None
        tmp_41 = tmp_40.transpose(1, 2);  tmp_40 = None
        tmp_42 = torch.cat((tmp_34, tmp_41), dim = 1);  tmp_34 = tmp_41 = None
        tmp_43 = torch.nn.functional.layer_norm(tmp_42, (216,), w_17, w_16, 1e-06);  w_17 = w_16 = None
        linear_1 = torch.nn.functional.linear(tmp_43, w_15, w_14);  tmp_43 = w_15 = w_14 = None
        tmp_45 = linear_1.reshape(1, 197, 3, 8, 27);  linear_1 = None
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
        tmp_58 = tmp_57.reshape(1, 216, 14, 14);  tmp_57 = None
        split = torch.functional.split(tmp_58, [54, 81, 81], dim = 1);  tmp_58 = None
        tmp_60 = split[0]
        tmp_61 = split[1]
        tmp_62 = split[2];  split = None
        conv2d_2 = torch.conv2d(tmp_60, w_9, w_8, (1, 1), (1, 1), (1, 1), 54);  tmp_60 = w_9 = w_8 = None
        conv2d_3 = torch.conv2d(tmp_61, w_11, w_10, (1, 1), (2, 2), (1, 1), 81);  tmp_61 = w_11 = w_10 = None
        conv2d_4 = torch.conv2d(tmp_62, w_13, w_12, (1, 1), (3, 3), (1, 1), 81);  tmp_62 = w_13 = w_12 = None
        tmp_66 = torch.cat([conv2d_2, conv2d_3, conv2d_4], dim = 1);  conv2d_2 = conv2d_3 = conv2d_4 = None
        tmp_67 = tmp_66.reshape(1, 8, 27, 196);  tmp_66 = None
        tmp_68 = tmp_67.transpose(-1, -2);  tmp_67 = None
        tmp_69 = tmp_55 * tmp_68;  tmp_55 = tmp_68 = None
        tmp_70 = torch.nn.functional.pad(tmp_69, (0, 0, 1, 0, 0, 0), 'constant', None);  tmp_69 = None
        tmp_71 = 0.19245008972987526 * matmul_1;  matmul_1 = None
        tmp_72 = tmp_71 + tmp_70;  tmp_71 = tmp_70 = None
        tmp_73 = tmp_72.transpose(1, 2);  tmp_72 = None
        tmp_74 = tmp_73.reshape(1, 197, 216);  tmp_73 = None
        return (tmp_23, tmp_42, tmp_74)
        
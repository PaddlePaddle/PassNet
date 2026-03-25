import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15):
        tmp_14 = torch.nn.functional.gelu(in_15, approximate = 'none');  in_15 = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, 0.0, False, False);  tmp_14 = None
        to = tmp_15.to(torch.float16);  tmp_15 = None
        linear = torch.nn.functional.linear(to, in_9, in_8);  to = in_9 = in_8 = None
        tmp_17 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_18 = in_14 + tmp_17;  in_14 = tmp_17 = None
        tmp_19 = tmp_18[(slice(None, None, None), slice(None, 1, None))]
        tmp_20 = tmp_18[(slice(None, None, None), slice(1, None, None))];  tmp_18 = None
        tmp_21 = tmp_20.transpose(1, 2);  tmp_20 = None
        tmp_22 = tmp_21.view(1, 256, 48, 48);  tmp_21 = None
        to_1 = tmp_22.to(torch.float16)
        conv2d = torch.conv2d(to_1, in_1, in_0, (1, 1), (1, 1), (1, 1), 256);  to_1 = in_1 = in_0 = None
        tmp_24 = conv2d + tmp_22;  conv2d = tmp_22 = None
        tmp_25 = tmp_24.flatten(2);  tmp_24 = None
        tmp_26 = tmp_25.transpose(1, 2);  tmp_25 = None
        tmp_27 = torch.cat((tmp_19, tmp_26), dim = 1);  tmp_19 = tmp_26 = None
        tmp_28 = torch.nn.functional.layer_norm(tmp_27, (256,), in_13, in_12, 1e-06);  in_13 = in_12 = None
        to_2 = tmp_28.to(torch.float16);  tmp_28 = None
        linear_1 = torch.nn.functional.linear(to_2, in_11, in_10);  to_2 = in_11 = in_10 = None
        tmp_30 = linear_1.reshape(1, 2305, 3, 8, 32);  linear_1 = None
        tmp_31 = tmp_30.permute(2, 0, 3, 1, 4);  tmp_30 = None
        unbind = tmp_31.unbind(0);  tmp_31 = None
        tmp_33 = unbind[0]
        tmp_34 = unbind[1]
        tmp_35 = unbind[2];  unbind = None
        tmp_36 = tmp_34.softmax(dim = 2);  tmp_34 = None
        tmp_37 = tmp_36.transpose(-1, -2);  tmp_36 = None
        to_3 = tmp_37.to(torch.float16);  tmp_37 = None
        to_4 = tmp_35.to(torch.float16)
        matmul = to_3 @ to_4;  to_3 = to_4 = None
        to_5 = tmp_33.to(torch.float16)
        to_6 = matmul.to(torch.float16);  matmul = None
        matmul_1 = to_5 @ to_6;  to_5 = to_6 = None
        tmp_40 = tmp_33[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  tmp_33 = None
        tmp_41 = tmp_35[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  tmp_35 = None
        tmp_42 = tmp_41.transpose(-1, -2);  tmp_41 = None
        tmp_43 = tmp_42.reshape(1, 256, 48, 48);  tmp_42 = None
        split = torch.functional.split(tmp_43, [64, 96, 96], dim = 1);  tmp_43 = None
        tmp_45 = split[0]
        tmp_46 = split[1]
        tmp_47 = split[2];  split = None
        to_7 = tmp_45.to(torch.float16);  tmp_45 = None
        conv2d_1 = torch.conv2d(to_7, in_3, in_2, (1, 1), (1, 1), (1, 1), 64);  to_7 = in_3 = in_2 = None
        to_8 = tmp_46.to(torch.float16);  tmp_46 = None
        conv2d_2 = torch.conv2d(to_8, in_5, in_4, (1, 1), (2, 2), (1, 1), 96);  to_8 = in_5 = in_4 = None
        to_9 = tmp_47.to(torch.float16);  tmp_47 = None
        conv2d_3 = torch.conv2d(to_9, in_7, in_6, (1, 1), (3, 3), (1, 1), 96);  to_9 = in_7 = in_6 = None
        tmp_51 = torch.cat([conv2d_1, conv2d_2, conv2d_3], dim = 1);  conv2d_1 = conv2d_2 = conv2d_3 = None
        tmp_52 = tmp_51.reshape(1, 8, 32, 2304);  tmp_51 = None
        tmp_53 = tmp_52.transpose(-1, -2);  tmp_52 = None
        tmp_54 = tmp_40 * tmp_53;  tmp_40 = tmp_53 = None
        tmp_55 = torch.nn.functional.pad(tmp_54, (0, 0, 1, 0, 0, 0), 'constant', None);  tmp_54 = None
        tmp_56 = 0.1767766952966369 * matmul_1;  matmul_1 = None
        tmp_57 = tmp_56 + tmp_55;  tmp_56 = tmp_55 = None
        tmp_58 = tmp_57.transpose(1, 2);  tmp_57 = None
        tmp_59 = tmp_58.reshape(1, 2305, 256);  tmp_58 = None
        return (tmp_27, tmp_59)
        
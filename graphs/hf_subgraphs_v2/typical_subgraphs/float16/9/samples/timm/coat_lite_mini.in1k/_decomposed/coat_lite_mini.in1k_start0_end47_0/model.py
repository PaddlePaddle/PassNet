import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, in_0 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_3, w_2, (4, 4), (0, 0), (1, 1), 1);  in_0 = w_3 = w_2 = None
        tmp_19 = conv2d.flatten(2);  conv2d = None
        tmp_20 = tmp_19.transpose(1, 2);  tmp_19 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (64,), w_1, w_0, 1e-05);  tmp_20 = w_1 = w_0 = None
        tmp_22 = w_16.expand(1, -1, -1);  w_16 = None
        tmp_23 = torch.cat((tmp_22, tmp_21), dim = 1);  tmp_22 = tmp_21 = None
        tmp_24 = tmp_23[(slice(None, None, None), slice(None, 1, None))]
        tmp_25 = tmp_23[(slice(None, None, None), slice(1, None, None))];  tmp_23 = None
        tmp_26 = tmp_25.transpose(1, 2);  tmp_25 = None
        tmp_27 = tmp_26.view(1, 64, 56, 56);  tmp_26 = None
        conv2d_1 = torch.conv2d(tmp_27, w_5, w_4, (1, 1), (1, 1), (1, 1), 64);  w_5 = w_4 = None
        tmp_29 = conv2d_1 + tmp_27;  conv2d_1 = tmp_27 = None
        tmp_30 = tmp_29.flatten(2);  tmp_29 = None
        tmp_31 = tmp_30.transpose(1, 2);  tmp_30 = None
        tmp_32 = torch.cat((tmp_24, tmp_31), dim = 1);  tmp_24 = tmp_31 = None
        tmp_33 = torch.nn.functional.layer_norm(tmp_32, (64,), w_15, w_14, 1e-06);  w_15 = w_14 = None
        linear = torch.nn.functional.linear(tmp_33, w_13, w_12);  tmp_33 = w_13 = w_12 = None
        tmp_35 = linear.reshape(1, 3137, 3, 8, 8);  linear = None
        tmp_36 = tmp_35.permute(2, 0, 3, 1, 4);  tmp_35 = None
        unbind = tmp_36.unbind(0);  tmp_36 = None
        tmp_38 = unbind[0]
        tmp_39 = unbind[1]
        tmp_40 = unbind[2];  unbind = None
        tmp_41 = tmp_39.softmax(dim = 2);  tmp_39 = None
        tmp_42 = tmp_41.transpose(-1, -2);  tmp_41 = None
        matmul = tmp_42 @ tmp_40;  tmp_42 = None
        matmul_1 = tmp_38 @ matmul;  matmul = None
        tmp_45 = tmp_38[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  tmp_38 = None
        tmp_46 = tmp_40[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  tmp_40 = None
        tmp_47 = tmp_46.transpose(-1, -2);  tmp_46 = None
        tmp_48 = tmp_47.reshape(1, 64, 56, 56);  tmp_47 = None
        split = torch.functional.split(tmp_48, [16, 24, 24], dim = 1);  tmp_48 = None
        tmp_50 = split[0]
        tmp_51 = split[1]
        tmp_52 = split[2];  split = None
        conv2d_2 = torch.conv2d(tmp_50, w_7, w_6, (1, 1), (1, 1), (1, 1), 16);  tmp_50 = w_7 = w_6 = None
        conv2d_3 = torch.conv2d(tmp_51, w_9, w_8, (1, 1), (2, 2), (1, 1), 24);  tmp_51 = w_9 = w_8 = None
        conv2d_4 = torch.conv2d(tmp_52, w_11, w_10, (1, 1), (3, 3), (1, 1), 24);  tmp_52 = w_11 = w_10 = None
        tmp_56 = torch.cat([conv2d_2, conv2d_3, conv2d_4], dim = 1);  conv2d_2 = conv2d_3 = conv2d_4 = None
        tmp_57 = tmp_56.reshape(1, 8, 8, 3136);  tmp_56 = None
        tmp_58 = tmp_57.transpose(-1, -2);  tmp_57 = None
        tmp_59 = tmp_45 * tmp_58;  tmp_45 = tmp_58 = None
        tmp_60 = torch.nn.functional.pad(tmp_59, (0, 0, 1, 0, 0, 0), 'constant', None);  tmp_59 = None
        tmp_61 = 0.3535533905932738 * matmul_1;  matmul_1 = None
        tmp_62 = tmp_61 + tmp_60;  tmp_61 = tmp_60 = None
        tmp_63 = tmp_62.transpose(1, 2);  tmp_62 = None
        tmp_64 = tmp_63.reshape(1, 3137, 64);  tmp_63 = None
        return (tmp_32, tmp_64)
        
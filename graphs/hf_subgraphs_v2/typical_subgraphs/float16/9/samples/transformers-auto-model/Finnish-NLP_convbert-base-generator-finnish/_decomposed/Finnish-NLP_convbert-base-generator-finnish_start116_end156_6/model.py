import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, in_0, in_1, in_2):
        linear = torch.nn.functional.linear(in_1, w_10, w_9);  w_10 = w_9 = None
        tmp_12 = in_1.transpose(1, 2)
        conv1d = torch.conv1d(tmp_12, w_4, None, (1,), (4,), (1,), 256);  tmp_12 = w_4 = None
        conv1d_1 = torch.conv1d(conv1d, w_5, None, (1,), (0,), (1,), 1);  conv1d = w_5 = None
        conv1d_1 += w_6;  tmp_15 = conv1d_1;  conv1d_1 = w_6 = None
        tmp_16 = tmp_15.transpose(1, 2);  tmp_15 = None
        linear_1 = torch.nn.functional.linear(in_1, w_8, w_7);  w_8 = w_7 = None
        tmp_18 = linear_1.view(1, -1, 2, 64)
        tmp_19 = tmp_18.transpose(1, 2);  tmp_18 = None
        tmp_20 = in_2.view(1, -1, 2, 64);  in_2 = None
        tmp_21 = tmp_20.transpose(1, 2);  tmp_20 = None
        tmp_22 = linear.view(1, -1, 2, 64);  linear = None
        tmp_23 = tmp_22.transpose(1, 2);  tmp_22 = None
        tmp_24 = torch.multiply(tmp_16, linear_1);  tmp_16 = linear_1 = None
        linear_2 = torch.nn.functional.linear(tmp_24, w_1, w_0);  tmp_24 = w_1 = w_0 = None
        tmp_26 = torch.reshape(linear_2, [-1, 9, 1]);  linear_2 = None
        tmp_27 = torch.softmax(tmp_26, dim = 1);  tmp_26 = None
        linear_3 = torch.nn.functional.linear(in_1, w_3, w_2);  in_1 = w_3 = w_2 = None
        tmp_29 = torch.reshape(linear_3, [1, -1, 128]);  linear_3 = None
        tmp_30 = tmp_29.transpose(1, 2);  tmp_29 = None
        tmp_31 = tmp_30.contiguous();  tmp_30 = None
        tmp_32 = tmp_31.unsqueeze(-1);  tmp_31 = None
        tmp_33 = torch.nn.functional.unfold(tmp_32, kernel_size = [9, 1], dilation = 1, padding = [4, 0], stride = 1);  tmp_32 = None
        tmp_34 = tmp_33.transpose(1, 2);  tmp_33 = None
        tmp_35 = tmp_34.reshape(1, -1, 128, 9);  tmp_34 = None
        tmp_36 = torch.reshape(tmp_35, [-1, 64, 9]);  tmp_35 = None
        matmul = torch.matmul(tmp_36, tmp_27);  tmp_36 = tmp_27 = None
        tmp_38 = torch.reshape(matmul, [-1, 128]);  matmul = None
        tmp_39 = tmp_21.transpose(-1, -2);  tmp_21 = None
        matmul_1 = torch.matmul(tmp_19, tmp_39);  tmp_19 = tmp_39 = None
        tmp_41 = matmul_1 / 8.0;  matmul_1 = None
        tmp_42 = tmp_41 + in_0;  tmp_41 = in_0 = None
        tmp_43 = torch.nn.functional.softmax(tmp_42, dim = -1);  tmp_42 = None
        tmp_44 = torch.nn.functional.dropout(tmp_43, 0.1, False, False);  tmp_43 = None
        matmul_2 = torch.matmul(tmp_44, tmp_23);  tmp_44 = tmp_23 = None
        tmp_46 = matmul_2.permute(0, 2, 1, 3);  matmul_2 = None
        tmp_47 = tmp_46.contiguous();  tmp_46 = None
        tmp_48 = torch.reshape(tmp_38, [1, -1, 2, 64]);  tmp_38 = None
        tmp_49 = torch.cat([tmp_47, tmp_48], 2);  tmp_47 = tmp_48 = None
        tmp_50 = tmp_49.view(1, 19, 256);  tmp_49 = None
        return (tmp_50,)
        
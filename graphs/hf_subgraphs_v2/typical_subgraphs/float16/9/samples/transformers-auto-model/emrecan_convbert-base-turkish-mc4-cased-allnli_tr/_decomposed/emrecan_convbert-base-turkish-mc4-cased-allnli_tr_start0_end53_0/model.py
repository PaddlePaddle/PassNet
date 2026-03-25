import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, in_2 : torch.Tensor):
        tmp_22 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_23 = tmp_22.to(dtype = torch.float32);  tmp_22 = None
        tmp_24 = 1.0 - tmp_23;  tmp_23 = None
        tmp_25 = tmp_24 * -3.4028234663852886e+38;  tmp_24 = None
        tmp_26 = w_0[(slice(None, None, None), slice(None, 23, None))];  w_0 = None
        tmp_27 = torch.nn.functional.embedding(in_1, w_5, 0, None, 2.0, False, False);  in_1 = w_5 = None
        tmp_28 = torch.nn.functional.embedding(tmp_26, w_3, None, None, 2.0, False, False);  tmp_26 = w_3 = None
        tmp_29 = torch.nn.functional.embedding(in_2, w_4, None, None, 2.0, False, False);  in_2 = w_4 = None
        tmp_30 = tmp_27 + tmp_28;  tmp_27 = tmp_28 = None
        tmp_31 = tmp_30 + tmp_29;  tmp_30 = tmp_29 = None
        tmp_32 = torch.nn.functional.layer_norm(tmp_31, (768,), w_2, w_1, 1e-12);  tmp_31 = w_2 = w_1 = None
        tmp_33 = torch.nn.functional.dropout(tmp_32, 0.1, False, False);  tmp_32 = None
        linear = torch.nn.functional.linear(tmp_33, w_14, w_13);  w_14 = w_13 = None
        linear_1 = torch.nn.functional.linear(tmp_33, w_18, w_17);  w_18 = w_17 = None
        tmp_36 = tmp_33.transpose(1, 2)
        conv1d = torch.conv1d(tmp_36, w_10, None, (1,), (4,), (1,), 768);  tmp_36 = w_10 = None
        conv1d_1 = torch.conv1d(conv1d, w_11, None, (1,), (0,), (1,), 1);  conv1d = w_11 = None
        conv1d_1 += w_12;  tmp_39 = conv1d_1;  conv1d_1 = w_12 = None
        tmp_40 = tmp_39.transpose(1, 2);  tmp_39 = None
        linear_2 = torch.nn.functional.linear(tmp_33, w_16, w_15);  w_16 = w_15 = None
        tmp_42 = linear_2.view(1, -1, 6, 64)
        tmp_43 = tmp_42.transpose(1, 2);  tmp_42 = None
        tmp_44 = linear.view(1, -1, 6, 64);  linear = None
        tmp_45 = tmp_44.transpose(1, 2);  tmp_44 = None
        tmp_46 = linear_1.view(1, -1, 6, 64);  linear_1 = None
        tmp_47 = tmp_46.transpose(1, 2);  tmp_46 = None
        tmp_48 = torch.multiply(tmp_40, linear_2);  tmp_40 = linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_48, w_7, w_6);  tmp_48 = w_7 = w_6 = None
        tmp_50 = torch.reshape(linear_3, [-1, 9, 1]);  linear_3 = None
        tmp_51 = torch.softmax(tmp_50, dim = 1);  tmp_50 = None
        linear_4 = torch.nn.functional.linear(tmp_33, w_9, w_8);  w_9 = w_8 = None
        tmp_53 = torch.reshape(linear_4, [1, -1, 384]);  linear_4 = None
        tmp_54 = tmp_53.transpose(1, 2);  tmp_53 = None
        tmp_55 = tmp_54.contiguous();  tmp_54 = None
        tmp_56 = tmp_55.unsqueeze(-1);  tmp_55 = None
        tmp_57 = torch.nn.functional.unfold(tmp_56, kernel_size = [9, 1], dilation = 1, padding = [4, 0], stride = 1);  tmp_56 = None
        tmp_58 = tmp_57.transpose(1, 2);  tmp_57 = None
        tmp_59 = tmp_58.reshape(1, -1, 384, 9);  tmp_58 = None
        tmp_60 = torch.reshape(tmp_59, [-1, 64, 9]);  tmp_59 = None
        matmul = torch.matmul(tmp_60, tmp_51);  tmp_60 = tmp_51 = None
        tmp_62 = torch.reshape(matmul, [-1, 384]);  matmul = None
        tmp_63 = tmp_45.transpose(-1, -2);  tmp_45 = None
        matmul_1 = torch.matmul(tmp_43, tmp_63);  tmp_43 = tmp_63 = None
        tmp_65 = matmul_1 / 8.0;  matmul_1 = None
        tmp_66 = tmp_65 + tmp_25;  tmp_65 = None
        tmp_67 = torch.nn.functional.softmax(tmp_66, dim = -1);  tmp_66 = None
        tmp_68 = torch.nn.functional.dropout(tmp_67, 0.1, False, False);  tmp_67 = None
        to_11 = tmp_68.to(torch.float16);  tmp_68 = None
        matmul_2 = torch.matmul(to_11, tmp_47);  to_11 = tmp_47 = None
        tmp_70 = matmul_2.permute(0, 2, 1, 3);  matmul_2 = None
        tmp_71 = tmp_70.contiguous();  tmp_70 = None
        tmp_72 = torch.reshape(tmp_62, [1, -1, 6, 64]);  tmp_62 = None
        tmp_73 = torch.cat([tmp_71, tmp_72], 2);  tmp_71 = tmp_72 = None
        tmp_74 = tmp_73.view(1, 23, 768);  tmp_73 = None
        return (tmp_74, tmp_33, tmp_25)
        
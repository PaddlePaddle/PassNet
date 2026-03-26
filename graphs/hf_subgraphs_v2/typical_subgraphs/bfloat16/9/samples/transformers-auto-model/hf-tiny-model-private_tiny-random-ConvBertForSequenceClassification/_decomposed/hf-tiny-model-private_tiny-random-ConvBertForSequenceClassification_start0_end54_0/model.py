import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, w_20 : torch.Tensor, in_2 : torch.Tensor):
        tmp_24 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_25 = tmp_24.to(dtype = torch.float32);  tmp_24 = None
        tmp_26 = 1.0 - tmp_25;  tmp_25 = None
        tmp_27 = tmp_26 * -3.4028234663852886e+38;  tmp_26 = None
        tmp_28 = w_0[(slice(None, None, None), slice(None, 45, None))];  w_0 = None
        tmp_29 = torch.nn.functional.embedding(in_1, w_5, 0, None, 2.0, False, False);  in_1 = w_5 = None
        tmp_30 = torch.nn.functional.embedding(tmp_28, w_3, None, None, 2.0, False, False);  tmp_28 = w_3 = None
        tmp_31 = torch.nn.functional.embedding(in_2, w_4, None, None, 2.0, False, False);  in_2 = w_4 = None
        tmp_32 = tmp_29 + tmp_30;  tmp_29 = tmp_30 = None
        tmp_33 = tmp_32 + tmp_31;  tmp_32 = tmp_31 = None
        tmp_34 = torch.nn.functional.layer_norm(tmp_33, (768,), w_2, w_1, 1e-12);  tmp_33 = w_2 = w_1 = None
        tmp_35 = torch.nn.functional.dropout(tmp_34, 0.1, False, False);  tmp_34 = None
        linear = torch.nn.functional.linear(tmp_35, w_7, w_6);  tmp_35 = w_7 = w_6 = None
        linear_1 = torch.nn.functional.linear(linear, w_16, w_15);  w_16 = w_15 = None
        linear_2 = torch.nn.functional.linear(linear, w_20, w_19);  w_20 = w_19 = None
        tmp_39 = linear.transpose(1, 2)
        conv1d = torch.conv1d(tmp_39, w_12, None, (1,), (4,), (1,), 32);  tmp_39 = w_12 = None
        conv1d_1 = torch.conv1d(conv1d, w_13, None, (1,), (0,), (1,), 1);  conv1d = w_13 = None
        conv1d_1 += w_14;  tmp_42 = conv1d_1;  conv1d_1 = w_14 = None
        tmp_43 = tmp_42.transpose(1, 2);  tmp_42 = None
        linear_3 = torch.nn.functional.linear(linear, w_18, w_17);  w_18 = w_17 = None
        tmp_45 = linear_3.view(1, -1, 2, 8)
        tmp_46 = tmp_45.transpose(1, 2);  tmp_45 = None
        tmp_47 = linear_1.view(1, -1, 2, 8);  linear_1 = None
        tmp_48 = tmp_47.transpose(1, 2);  tmp_47 = None
        tmp_49 = linear_2.view(1, -1, 2, 8);  linear_2 = None
        tmp_50 = tmp_49.transpose(1, 2);  tmp_49 = None
        tmp_51 = torch.multiply(tmp_43, linear_3);  tmp_43 = linear_3 = None
        linear_4 = torch.nn.functional.linear(tmp_51, w_9, w_8);  tmp_51 = w_9 = w_8 = None
        tmp_53 = torch.reshape(linear_4, [-1, 9, 1]);  linear_4 = None
        tmp_54 = torch.softmax(tmp_53, dim = 1);  tmp_53 = None
        linear_5 = torch.nn.functional.linear(linear, w_11, w_10);  w_11 = w_10 = None
        tmp_56 = torch.reshape(linear_5, [1, -1, 16]);  linear_5 = None
        tmp_57 = tmp_56.transpose(1, 2);  tmp_56 = None
        tmp_58 = tmp_57.contiguous();  tmp_57 = None
        tmp_59 = tmp_58.unsqueeze(-1);  tmp_58 = None
        tmp_60 = torch.nn.functional.unfold(tmp_59, kernel_size = [9, 1], dilation = 1, padding = [4, 0], stride = 1);  tmp_59 = None
        tmp_61 = tmp_60.transpose(1, 2);  tmp_60 = None
        tmp_62 = tmp_61.reshape(1, -1, 16, 9);  tmp_61 = None
        tmp_63 = torch.reshape(tmp_62, [-1, 8, 9]);  tmp_62 = None
        matmul = torch.matmul(tmp_63, tmp_54);  tmp_63 = tmp_54 = None
        tmp_65 = torch.reshape(matmul, [-1, 16]);  matmul = None
        tmp_66 = tmp_48.transpose(-1, -2);  tmp_48 = None
        matmul_1 = torch.matmul(tmp_46, tmp_66);  tmp_46 = tmp_66 = None
        tmp_68 = matmul_1 / 2.8284271247461903;  matmul_1 = None
        tmp_69 = tmp_68 + tmp_27;  tmp_68 = None
        tmp_70 = torch.nn.functional.softmax(tmp_69, dim = -1);  tmp_69 = None
        tmp_71 = torch.nn.functional.dropout(tmp_70, 0.1, False, False);  tmp_70 = None
        to_12 = tmp_71.to(torch.bfloat16);  tmp_71 = None
        matmul_2 = torch.matmul(to_12, tmp_50);  to_12 = tmp_50 = None
        tmp_73 = matmul_2.permute(0, 2, 1, 3);  matmul_2 = None
        tmp_74 = tmp_73.contiguous();  tmp_73 = None
        tmp_75 = torch.reshape(tmp_65, [1, -1, 2, 8]);  tmp_65 = None
        tmp_76 = torch.cat([tmp_74, tmp_75], 2);  tmp_74 = tmp_75 = None
        tmp_77 = tmp_76.view(1, 45, 32);  tmp_76 = None
        return (tmp_77, tmp_27, linear)
        
import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor):
        tmp_11 = in_0.__eq__(1)
        tmp_12 = tmp_11.to(torch.float32);  tmp_11 = None
        tmp_12 *= -3.4028234663852886e+38;  tmp_13 = tmp_12;  tmp_12 = None
        tmp_14 = tmp_13.unsqueeze(1);  tmp_13 = None
        tmp_15 = tmp_14.unsqueeze(1);  tmp_14 = None
        tmp_16 = torch.nn.functional.embedding(in_0, w_3, 1, None, 2.0, False, False);  in_0 = w_3 = None
        tmp_17 = torch.ones((1, 15), dtype = torch.int64, device = device(type='cuda', index=0))
        tmp_18 = torch.cumsum(tmp_17, dim = 1)
        tmp_19 = tmp_18 - tmp_17;  tmp_18 = tmp_17 = None
        tmp_19 += 2;  tmp_20 = tmp_19;  tmp_19 = None
        tmp_21 = torch.nn.functional.embedding(tmp_20, w_2, 1, None, 2.0, False, False);  tmp_20 = w_2 = None
        tmp_22 = tmp_16 + tmp_21;  tmp_16 = tmp_21 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (32,), w_1, w_0, 1e-05);  tmp_22 = w_1 = w_0 = None
        tmp_24 = torch.nn.functional.dropout(tmp_23, 0.1, False, False);  tmp_23 = None
        linear = torch.nn.functional.linear(tmp_24, w_7, w_6);  w_7 = w_6 = None
        linear_1 = torch.nn.functional.linear(tmp_24, w_5, w_4);  w_5 = w_4 = None
        tmp_27 = linear_1.view((1, 15, 4, 8));  linear_1 = None
        tmp_28 = tmp_27.permute(0, 2, 1, 3);  tmp_27 = None
        linear_2 = torch.nn.functional.linear(tmp_24, w_9, w_8);  w_9 = w_8 = None
        tmp_30 = linear_2.view((1, 15, 4, 8));  linear_2 = None
        tmp_31 = tmp_30.permute(0, 2, 1, 3);  tmp_30 = None
        tmp_32 = linear.view((1, 15, 4, 8));  linear = None
        tmp_33 = tmp_32.permute(0, 2, 1, 3);  tmp_32 = None
        tmp_34 = tmp_28.transpose(-1, -2);  tmp_28 = None
        matmul = torch.matmul(tmp_33, tmp_34);  tmp_33 = tmp_34 = None
        tmp_36 = matmul / 2.8284271247461903;  matmul = None
        tmp_37 = tmp_36 + tmp_15;  tmp_36 = None
        tmp_38 = torch.nn.functional.softmax(tmp_37, dim = -1);  tmp_37 = None
        tmp_39 = torch.nn.functional.dropout(tmp_38, 0.1, False, False);  tmp_38 = None
        to_5 = tmp_39.to(torch.float16);  tmp_39 = None
        matmul_1 = torch.matmul(to_5, tmp_31);  to_5 = tmp_31 = None
        tmp_41 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_42 = tmp_41.contiguous();  tmp_41 = None
        tmp_43 = tmp_42.view((1, 15, 32));  tmp_42 = None
        return (tmp_43, tmp_24, tmp_15)
        
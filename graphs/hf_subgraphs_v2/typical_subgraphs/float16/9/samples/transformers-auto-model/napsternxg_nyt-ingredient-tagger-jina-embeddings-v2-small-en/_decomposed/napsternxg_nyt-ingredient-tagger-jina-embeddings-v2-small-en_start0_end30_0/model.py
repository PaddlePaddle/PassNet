import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, in_2 : torch.Tensor):
        tmp_14 = in_0[(slice(None, None, None), None, None, slice(None, None, None))];  in_0 = None
        tmp_15 = tmp_14.to(dtype = torch.float32);  tmp_14 = None
        tmp_16 = 1.0 - tmp_15;  tmp_15 = None
        tmp_17 = tmp_16 * -3.4028234663852886e+38;  tmp_16 = None
        tmp_18 = torch.nn.functional.embedding(in_1, w_3, 0, None, 2.0, False, False);  in_1 = w_3 = None
        tmp_19 = torch.nn.functional.embedding(in_2, w_2, None, None, 2.0, False, False);  in_2 = w_2 = None
        tmp_20 = tmp_18 + tmp_19;  tmp_18 = tmp_19 = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (512,), w_1, w_0, 1e-12);  tmp_20 = w_1 = w_0 = None
        tmp_22 = torch.nn.functional.dropout(tmp_21, 0.1, False, False);  tmp_21 = None
        tmp_23 = w_4[(slice(None, None, None), slice(None, None, None), slice(None, 11, None), slice(None, 11, None))];  w_4 = None
        linear = torch.nn.functional.linear(tmp_22, w_8, w_7);  w_8 = w_7 = None
        linear_1 = torch.nn.functional.linear(tmp_22, w_6, w_5);  w_6 = w_5 = None
        tmp_26 = linear_1.view((1, 11, 8, 64));  linear_1 = None
        tmp_27 = tmp_26.permute(0, 2, 1, 3);  tmp_26 = None
        linear_2 = torch.nn.functional.linear(tmp_22, w_10, w_9);  w_10 = w_9 = None
        tmp_29 = linear_2.view((1, 11, 8, 64));  linear_2 = None
        tmp_30 = tmp_29.permute(0, 2, 1, 3);  tmp_29 = None
        tmp_31 = linear.view((1, 11, 8, 64));  linear = None
        tmp_32 = tmp_31.permute(0, 2, 1, 3);  tmp_31 = None
        tmp_33 = tmp_27.transpose(-1, -2);  tmp_27 = None
        matmul = torch.matmul(tmp_32, tmp_33);  tmp_32 = tmp_33 = None
        tmp_35 = matmul / 8.0;  matmul = None
        tmp_36 = tmp_35 + tmp_17;  tmp_35 = None
        tmp_37 = tmp_36 + tmp_23;  tmp_36 = None
        tmp_38 = torch.nn.functional.softmax(tmp_37, dim = -1);  tmp_37 = None
        tmp_39 = torch.nn.functional.dropout(tmp_38, 0.0, False, False);  tmp_38 = None
        to_5 = tmp_39.to(torch.float16);  tmp_39 = None
        matmul_1 = torch.matmul(to_5, tmp_30);  to_5 = tmp_30 = None
        tmp_41 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_42 = tmp_41.contiguous();  tmp_41 = None
        tmp_43 = tmp_42.view((1, 11, 512));  tmp_42 = None
        return (tmp_23, tmp_43, tmp_22, tmp_17)
        
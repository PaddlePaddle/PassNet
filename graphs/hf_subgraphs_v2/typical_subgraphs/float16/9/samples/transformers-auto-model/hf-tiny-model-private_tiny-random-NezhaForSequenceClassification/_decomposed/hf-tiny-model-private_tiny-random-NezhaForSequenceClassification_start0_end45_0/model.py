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
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (32,), w_1, w_0, 1e-12);  tmp_20 = w_1 = w_0 = None
        tmp_22 = torch.nn.functional.dropout(tmp_21, 0.1, False, False);  tmp_21 = None
        linear = torch.nn.functional.linear(tmp_22, w_7, w_6);  w_7 = w_6 = None
        linear_1 = torch.nn.functional.linear(tmp_22, w_5, w_4);  w_5 = w_4 = None
        tmp_25 = linear_1.view((1, 45, 4, 8));  linear_1 = None
        tmp_26 = tmp_25.permute(0, 2, 1, 3);  tmp_25 = None
        linear_2 = torch.nn.functional.linear(tmp_22, w_10, w_9);  w_10 = w_9 = None
        tmp_28 = linear_2.view((1, 45, 4, 8));  linear_2 = None
        tmp_29 = tmp_28.permute(0, 2, 1, 3);  tmp_28 = None
        tmp_30 = linear.view((1, 45, 4, 8));  linear = None
        tmp_31 = tmp_30.permute(0, 2, 1, 3);  tmp_30 = None
        tmp_32 = tmp_26.transpose(-1, -2);  tmp_26 = None
        matmul = torch.matmul(tmp_31, tmp_32);  tmp_32 = None
        tmp_34 = w_8[(slice(None, 45, None), slice(None, 45, None), slice(None, None, None))]
        tmp_35 = tmp_31.permute(2, 0, 1, 3);  tmp_31 = None
        tmp_36 = tmp_35.contiguous();  tmp_35 = None
        tmp_37 = tmp_36.view(45, 4, 8);  tmp_36 = None
        tmp_38 = tmp_34.permute(0, 2, 1);  tmp_34 = None
        matmul_1 = torch.matmul(tmp_37, tmp_38);  tmp_37 = tmp_38 = None
        tmp_40 = matmul_1.view(45, 1, 4, 45);  matmul_1 = None
        tmp_41 = tmp_40.permute(1, 2, 0, 3);  tmp_40 = None
        tmp_42 = matmul + tmp_41;  matmul = tmp_41 = None
        tmp_43 = tmp_42 / 2.8284271247461903;  tmp_42 = None
        tmp_44 = tmp_43 + tmp_17;  tmp_43 = None
        tmp_45 = torch.nn.functional.softmax(tmp_44, dim = -1);  tmp_44 = None
        tmp_46 = torch.nn.functional.dropout(tmp_45, 0.1, False, False);  tmp_45 = None
        to_7 = tmp_46.to(torch.float16)
        matmul_2 = torch.matmul(to_7, tmp_29);  to_7 = tmp_29 = None
        tmp_48 = w_8[(slice(None, 45, None), slice(None, 45, None), slice(None, None, None))];  w_8 = None
        tmp_49 = tmp_46.permute(2, 0, 1, 3);  tmp_46 = None
        tmp_50 = tmp_49.contiguous();  tmp_49 = None
        tmp_51 = tmp_50.view(45, 4, 45);  tmp_50 = None
        to_9 = tmp_51.to(torch.float16);  tmp_51 = None
        matmul_3 = torch.matmul(to_9, tmp_48);  to_9 = tmp_48 = None
        tmp_53 = matmul_3.view(45, 1, 4, 8);  matmul_3 = None
        tmp_54 = tmp_53.permute(1, 2, 0, 3);  tmp_53 = None
        tmp_55 = matmul_2 + tmp_54;  matmul_2 = tmp_54 = None
        tmp_56 = tmp_55.permute(0, 2, 1, 3);  tmp_55 = None
        tmp_57 = tmp_56.contiguous();  tmp_56 = None
        tmp_58 = tmp_57.view((1, 45, 32));  tmp_57 = None
        return (tmp_58, tmp_22, tmp_17)
        
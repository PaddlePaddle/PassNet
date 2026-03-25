import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, in_0, in_1, in_2):
        linear = torch.nn.functional.linear(in_1, w_1, w_0);  w_1 = w_0 = None
        tmp_6 = linear.view((1, 45, 4, 8));  linear = None
        tmp_7 = tmp_6.permute(0, 2, 1, 3);  tmp_6 = None
        linear_1 = torch.nn.functional.linear(in_1, w_4, w_3);  in_1 = w_4 = w_3 = None
        tmp_9 = linear_1.view((1, 45, 4, 8));  linear_1 = None
        tmp_10 = tmp_9.permute(0, 2, 1, 3);  tmp_9 = None
        tmp_11 = in_2.view((1, 45, 4, 8));  in_2 = None
        tmp_12 = tmp_11.permute(0, 2, 1, 3);  tmp_11 = None
        tmp_13 = tmp_7.transpose(-1, -2);  tmp_7 = None
        matmul = torch.matmul(tmp_12, tmp_13);  tmp_13 = None
        tmp_15 = w_2[(slice(None, 45, None), slice(None, 45, None), slice(None, None, None))]
        tmp_16 = tmp_12.permute(2, 0, 1, 3);  tmp_12 = None
        tmp_17 = tmp_16.contiguous();  tmp_16 = None
        tmp_18 = tmp_17.view(45, 4, 8);  tmp_17 = None
        tmp_19 = tmp_15.permute(0, 2, 1);  tmp_15 = None
        matmul_1 = torch.matmul(tmp_18, tmp_19);  tmp_18 = tmp_19 = None
        tmp_21 = matmul_1.view(45, 1, 4, 45);  matmul_1 = None
        tmp_22 = tmp_21.permute(1, 2, 0, 3);  tmp_21 = None
        tmp_23 = matmul + tmp_22;  matmul = tmp_22 = None
        tmp_24 = tmp_23 / 2.8284271247461903;  tmp_23 = None
        tmp_25 = tmp_24 + in_0;  tmp_24 = in_0 = None
        tmp_26 = torch.nn.functional.softmax(tmp_25, dim = -1);  tmp_25 = None
        tmp_27 = torch.nn.functional.dropout(tmp_26, 0.1, False, False);  tmp_26 = None
        matmul_2 = torch.matmul(tmp_27, tmp_10);  tmp_10 = None
        tmp_29 = w_2[(slice(None, 45, None), slice(None, 45, None), slice(None, None, None))];  w_2 = None
        tmp_30 = tmp_27.permute(2, 0, 1, 3);  tmp_27 = None
        tmp_31 = tmp_30.contiguous();  tmp_30 = None
        tmp_32 = tmp_31.view(45, 4, 45);  tmp_31 = None
        matmul_3 = torch.matmul(tmp_32, tmp_29);  tmp_32 = tmp_29 = None
        tmp_34 = matmul_3.view(45, 1, 4, 8);  matmul_3 = None
        tmp_35 = tmp_34.permute(1, 2, 0, 3);  tmp_34 = None
        tmp_36 = matmul_2 + tmp_35;  matmul_2 = tmp_35 = None
        tmp_37 = tmp_36.permute(0, 2, 1, 3);  tmp_36 = None
        tmp_38 = tmp_37.contiguous();  tmp_37 = None
        tmp_39 = tmp_38.view((1, 45, 32));  tmp_38 = None
        return (tmp_39,)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, in_0, in_1, in_2):
        linear = torch.nn.functional.linear(in_1, w_1, w_0);  w_1 = w_0 = None
        tmp_5 = linear.view((1, 15, 4, 8));  linear = None
        tmp_6 = tmp_5.permute(0, 2, 1, 3);  tmp_5 = None
        linear_1 = torch.nn.functional.linear(in_1, w_3, w_2);  in_1 = w_3 = w_2 = None
        tmp_8 = linear_1.view((1, 15, 4, 8));  linear_1 = None
        tmp_9 = tmp_8.permute(0, 2, 1, 3);  tmp_8 = None
        tmp_10 = in_2.view((1, 15, 4, 8));  in_2 = None
        tmp_11 = tmp_10.permute(0, 2, 1, 3);  tmp_10 = None
        tmp_12 = tmp_6.transpose(-1, -2);  tmp_6 = None
        matmul = torch.matmul(tmp_11, tmp_12);  tmp_11 = tmp_12 = None
        tmp_14 = matmul / 2.8284271247461903;  matmul = None
        tmp_15 = tmp_14 + in_0;  tmp_14 = in_0 = None
        tmp_16 = torch.nn.functional.softmax(tmp_15, dim = -1);  tmp_15 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.1, False, False);  tmp_16 = None
        matmul_1 = torch.matmul(tmp_17, tmp_9);  tmp_17 = tmp_9 = None
        tmp_19 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_20 = tmp_19.contiguous();  tmp_19 = None
        tmp_21 = tmp_20.view((1, 15, 32));  tmp_20 = None
        return (tmp_21,)
        
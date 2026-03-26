import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_0, w_1, w_0);  in_0 = w_1 = w_0 = None
        tmp_3 = linear.view(1, -1, 4, 8);  linear = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = in_2.transpose(-1, -2);  in_2 = None
        matmul = torch.matmul(in_3, tmp_5);  in_3 = tmp_5 = None
        tmp_7 = matmul / 2.8284271247461903;  matmul = None
        tmp_8 = tmp_7 + in_1;  tmp_7 = in_1 = None
        tmp_9 = torch.nn.functional.softmax(tmp_8, dim = -1);  tmp_8 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.1, False, False);  tmp_9 = None
        matmul_1 = torch.matmul(tmp_10, tmp_4);  tmp_10 = tmp_4 = None
        tmp_12 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_13 = tmp_12.contiguous();  tmp_12 = None
        tmp_14 = tmp_13.view(1, 11, 32);  tmp_13 = None
        return (tmp_14,)
        
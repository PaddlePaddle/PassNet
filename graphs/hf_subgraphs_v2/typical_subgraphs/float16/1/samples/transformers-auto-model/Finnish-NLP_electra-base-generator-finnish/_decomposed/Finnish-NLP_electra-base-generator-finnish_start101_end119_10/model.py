import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_4 = in_6.view(1, -1, 4, 64);  in_6 = None
        tmp_5 = tmp_4.transpose(1, 2);  tmp_4 = None
        linear = torch.nn.functional.linear(in_5, in_1, in_0);  in_1 = in_0 = None
        tmp_7 = linear.view(1, -1, 4, 64);  linear = None
        tmp_8 = tmp_7.transpose(1, 2);  tmp_7 = None
        linear_1 = torch.nn.functional.linear(in_5, in_3, in_2);  in_5 = in_3 = in_2 = None
        tmp_10 = linear_1.view(1, -1, 4, 64);  linear_1 = None
        tmp_11 = tmp_10.transpose(1, 2);  tmp_10 = None
        tmp_12 = tmp_8.transpose(-1, -2);  tmp_8 = None
        matmul = torch.matmul(tmp_5, tmp_12);  tmp_5 = tmp_12 = None
        tmp_14 = matmul / 8.0;  matmul = None
        tmp_15 = tmp_14 + in_4;  tmp_14 = in_4 = None
        tmp_16 = torch.nn.functional.softmax(tmp_15, dim = -1);  tmp_15 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.1, False, False);  tmp_16 = None
        matmul_1 = torch.matmul(tmp_17, tmp_11);  tmp_17 = tmp_11 = None
        tmp_19 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_20 = tmp_19.contiguous();  tmp_19 = None
        tmp_21 = tmp_20.view((1, 512, 256));  tmp_20 = None
        return (tmp_21,)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1, in_2, in_3, in_4):
        linear = torch.nn.functional.linear(in_0, w_1, w_0);  in_0 = w_1 = w_0 = None
        tmp_3 = linear.view(1, -1, 12, 64);  linear = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = in_2.transpose(-1, -2);  in_2 = None
        matmul = torch.matmul(in_3, tmp_5);  in_3 = tmp_5 = None
        tmp_7 = matmul / 8.0;  matmul = None
        tmp_7 += in_4;  tmp_8 = tmp_7;  tmp_7 = in_4 = None
        tmp_9 = tmp_8 + in_1;  tmp_8 = in_1 = None
        tmp_10 = torch.nn.functional.softmax(tmp_9, dim = -1);  tmp_9 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.1, False, False);  tmp_10 = None
        matmul_1 = torch.matmul(tmp_11, tmp_4);  tmp_11 = tmp_4 = None
        tmp_13 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_14 = tmp_13.contiguous();  tmp_13 = None
        tmp_15 = tmp_14.view(1, 11, 768);  tmp_14 = None
        return (tmp_15,)
        
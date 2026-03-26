import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        linear = torch.nn.functional.linear(in_3, in_1, in_0);  in_3 = in_1 = in_0 = None
        tmp_3 = linear.view(1, -1, 8, 64);  linear = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = in_2.transpose(-1, -2);  in_2 = None
        to = tmp_5.to(torch.bfloat16);  tmp_5 = None
        matmul = torch.matmul(in_4, to);  in_4 = to = None
        tmp_7 = matmul / 8.0;  matmul = None
        tmp_8 = torch.nn.functional.softmax(tmp_7, dim = -1);  tmp_7 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False);  tmp_8 = None
        to_1 = tmp_9.to(torch.bfloat16);  tmp_9 = None
        to_2 = tmp_4.to(torch.bfloat16);  tmp_4 = None
        matmul_1 = torch.matmul(to_1, to_2);  to_1 = to_2 = None
        tmp_11 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_12 = tmp_11.contiguous();  tmp_11 = None
        tmp_13 = tmp_12.view((1, 256, 512));  tmp_12 = None
        return (tmp_13,)
        
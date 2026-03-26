import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        tmp_3 = linear.view(8, -1, 12, 64);  linear = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = in_4.transpose(-1, -2);  in_4 = None
        matmul = torch.matmul(in_5, tmp_5);  in_5 = tmp_5 = None
        tmp_7 = matmul / 8.0;  matmul = None
        tmp_8 = tmp_7 + in_3;  tmp_7 = in_3 = None
        tmp_9 = torch.nn.functional.softmax(tmp_8, dim = -1);  tmp_8 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.1, False, False);  tmp_9 = None
        matmul_1 = torch.matmul(tmp_10, tmp_4);  tmp_10 = tmp_4 = None
        tmp_12 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_13 = tmp_12.contiguous();  tmp_12 = None
        tmp_14 = tmp_13.view(8, 256, 768);  tmp_13 = None
        return (tmp_14,)
        
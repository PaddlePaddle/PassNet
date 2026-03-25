import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1, in_2):
        linear = torch.nn.functional.linear(in_1, w_1, w_0);  in_1 = w_1 = w_0 = None
        tmp_3 = linear.view(4, -1, 4, 30);  linear = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = in_0.transpose(-1, -2);  in_0 = None
        matmul = torch.matmul(in_2, tmp_5);  in_2 = tmp_5 = None
        tmp_7 = matmul / 5.477225575051661;  matmul = None
        tmp_8 = torch.nn.functional.softmax(tmp_7, dim = -1);  tmp_7 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False);  tmp_8 = None
        matmul_1 = torch.matmul(tmp_9, tmp_4);  tmp_9 = tmp_4 = None
        tmp_11 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_12 = tmp_11.contiguous();  tmp_11 = None
        tmp_13 = tmp_12.view(4, 64, 120);  tmp_12 = None
        return (tmp_13,)
        
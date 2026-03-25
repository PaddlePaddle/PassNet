import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        linear = torch.nn.functional.linear(in_3, in_1, in_0);  in_3 = in_1 = in_0 = None
        tmp_3 = linear.view(1, -1, 8, 32);  linear = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = in_2.transpose(-1, -2);  in_2 = None
        matmul = torch.matmul(in_4, tmp_5);  in_4 = tmp_5 = None
        tmp_7 = matmul / 5.656854249492381;  matmul = None
        tmp_8 = torch.nn.functional.softmax(tmp_7, dim = -1);  tmp_7 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False);  tmp_8 = None
        matmul_1 = torch.matmul(tmp_9, tmp_4);  tmp_9 = tmp_4 = None
        tmp_11 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_12 = tmp_11.contiguous();  tmp_11 = None
        tmp_13 = tmp_12.view((1, 256, 256));  tmp_12 = None
        return (tmp_13,)
        
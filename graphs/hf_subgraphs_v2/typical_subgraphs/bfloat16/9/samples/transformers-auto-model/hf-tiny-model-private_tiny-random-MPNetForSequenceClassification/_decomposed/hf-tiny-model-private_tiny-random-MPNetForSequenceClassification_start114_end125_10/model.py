import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        tmp_0 = in_1.transpose(-1, -2);  in_1 = None
        matmul = torch.matmul(in_2, tmp_0);  in_2 = tmp_0 = None
        tmp_2 = matmul / 4.0;  matmul = None
        tmp_2 += in_4;  tmp_3 = tmp_2;  tmp_2 = in_4 = None
        tmp_4 = tmp_3 + in_0;  tmp_3 = in_0 = None
        tmp_5 = torch.nn.functional.softmax(tmp_4, dim = -1);  tmp_4 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.1, False, False);  tmp_5 = None
        matmul_1 = torch.matmul(tmp_6, in_3);  tmp_6 = in_3 = None
        tmp_8 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_9 = tmp_8.contiguous();  tmp_8 = None
        tmp_10 = tmp_9.view(1, 45, 64);  tmp_9 = None
        return (tmp_10,)
        
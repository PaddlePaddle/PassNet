import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_0 = in_1.transpose(-1, -2);  in_1 = None
        matmul = torch.matmul(in_2, tmp_0);  in_2 = tmp_0 = None
        tmp_2 = matmul / 8.0;  matmul = None
        tmp_3 = tmp_2 + in_0;  tmp_2 = in_0 = None
        tmp_4 = torch.nn.functional.softmax(tmp_3, dim = -1);  tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.1, False, False);  tmp_4 = None
        matmul_1 = torch.matmul(tmp_5, in_3);  tmp_5 = in_3 = None
        tmp_7 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_8 = tmp_7.contiguous();  tmp_7 = None
        tmp_9 = tmp_8.view((1, 14, 512));  tmp_8 = None
        return (tmp_9,)
        
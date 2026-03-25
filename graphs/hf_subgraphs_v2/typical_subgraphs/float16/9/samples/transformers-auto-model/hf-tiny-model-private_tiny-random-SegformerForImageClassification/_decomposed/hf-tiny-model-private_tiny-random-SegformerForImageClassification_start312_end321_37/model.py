import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_0 = in_0.transpose(-1, -2);  in_0 = None
        matmul = torch.matmul(in_1, tmp_0);  in_1 = tmp_0 = None
        tmp_2 = matmul / 4.0;  matmul = None
        tmp_3 = torch.nn.functional.softmax(tmp_2, dim = -1);  tmp_2 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, 0.1, False, False);  tmp_3 = None
        matmul_1 = torch.matmul(tmp_4, in_2);  tmp_4 = in_2 = None
        tmp_6 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_7 = tmp_6.contiguous();  tmp_6 = None
        tmp_8 = tmp_7.view((1, 4, 128));  tmp_7 = None
        return (tmp_8,)
        
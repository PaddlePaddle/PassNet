import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        matmul = torch.matmul(in_0, in_1);  in_0 = in_1 = None
        tmp_1 = matmul / 8.0;  matmul = None
        tmp_2 = torch.nn.functional.softmax(tmp_1, dim = -1);  tmp_1 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False);  tmp_2 = None
        matmul_1 = torch.matmul(tmp_3, in_2);  tmp_3 = in_2 = None
        tmp_5 = matmul_1.permute(0, 2, 1, 3);  matmul_1 = None
        tmp_6 = tmp_5.contiguous();  tmp_5 = None
        tmp_7 = tmp_6.view((32, 4096, 128));  tmp_6 = None
        return (tmp_7,)
        
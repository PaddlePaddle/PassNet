import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        matmul = in_0 @ in_1;  in_0 = in_1 = None
        tmp_1 = matmul * 0.125;  matmul = None
        tmp_2 = tmp_1.softmax(dim = -1);  tmp_1 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False);  tmp_2 = None
        matmul_1 = tmp_3 @ in_2;  tmp_3 = in_2 = None
        tmp_5 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_6 = tmp_5.reshape(1, 1, 512);  tmp_5 = None
        return (tmp_6,)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        matmul = in_1 @ in_0;  in_0 = None
        tmp_1 = matmul.permute(0, 3, 1, 2);  matmul = None
        tmp_2 = in_1.permute(0, 3, 1, 2);  in_1 = None
        tmp_3 = tmp_1.reshape(1, 20, 20, 256);  tmp_1 = None
        tmp_4 = tmp_3.permute(0, 3, 1, 2);  tmp_3 = None
        tmp_5 = tmp_4.contiguous();  tmp_4 = None
        return (tmp_2, tmp_5)
        
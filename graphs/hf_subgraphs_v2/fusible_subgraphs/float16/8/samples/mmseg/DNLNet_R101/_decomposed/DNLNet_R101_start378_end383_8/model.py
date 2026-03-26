import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        matmul = torch.matmul(in_2, in_0);  in_2 = in_0 = None
        tmp_1 = matmul.permute(0, 2, 1);  matmul = None
        tmp_2 = tmp_1.contiguous();  tmp_1 = None
        tmp_3 = tmp_2.reshape(64, 256, 1, 1);  tmp_2 = None
        tmp_4 = in_1 + tmp_3;  in_1 = tmp_3 = None
        return (tmp_4,)
        
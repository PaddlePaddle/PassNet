import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        matmul = torch.matmul(in_1, in_0);  in_1 = in_0 = None
        tmp_1 = matmul.permute(0, 2, 1);  matmul = None
        tmp_2 = tmp_1.contiguous();  tmp_1 = None
        tmp_3 = tmp_2.reshape(1, 256, 1, 1);  tmp_2 = None
        tmp_4 = in_2 + tmp_3;  in_2 = tmp_3 = None
        return (tmp_4,)
        
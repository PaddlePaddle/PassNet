import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        matmul = torch.matmul(in_1, in_2);  in_2 = None
        tmp_2 = in_0[(slice(None, 45, None), slice(None, 45, None), slice(None, None, None))];  in_0 = None
        tmp_3 = in_1.permute(2, 0, 1, 3);  in_1 = None
        tmp_4 = tmp_3.contiguous();  tmp_3 = None
        tmp_5 = tmp_4.view(45, 4, 8);  tmp_4 = None
        tmp_6 = tmp_2.permute(0, 2, 1);  tmp_2 = None
        return (matmul, tmp_6, tmp_5)
        
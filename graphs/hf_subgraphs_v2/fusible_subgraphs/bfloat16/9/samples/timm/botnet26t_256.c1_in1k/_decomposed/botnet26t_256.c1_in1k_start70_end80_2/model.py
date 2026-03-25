import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        split = torch.functional.split(in_0, [256, 256, 256], dim = 1);  in_0 = None
        tmp_1 = split[0]
        tmp_2 = split[1]
        tmp_3 = split[2];  split = None
        tmp_4 = tmp_1.reshape(4, 64, -1);  tmp_1 = None
        tmp_5 = tmp_4.transpose(-1, -2);  tmp_4 = None
        tmp_6 = tmp_2.reshape(4, 64, -1);  tmp_2 = None
        tmp_7 = tmp_3.reshape(4, 64, -1);  tmp_3 = None
        tmp_8 = tmp_7.transpose(-1, -2);  tmp_7 = None
        matmul = tmp_5 @ tmp_6;  tmp_6 = None
        return (matmul, tmp_5, tmp_8)
        
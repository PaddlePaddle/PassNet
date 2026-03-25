import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        matmul = in_1 @ in_0;  in_0 = None
        tmp_1 = in_1[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  in_1 = None
        tmp_2 = in_2[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  in_2 = None
        tmp_3 = tmp_2.transpose(-1, -2);  tmp_2 = None
        tmp_4 = tmp_3.reshape(1, 216, 28, 28);  tmp_3 = None
        split = torch.functional.split(tmp_4, [54, 81, 81], dim = 1);  tmp_4 = None
        tmp_6 = split[0]
        tmp_7 = split[1]
        tmp_8 = split[2];  split = None
        return (matmul, tmp_6, tmp_7, tmp_8, tmp_1)
        
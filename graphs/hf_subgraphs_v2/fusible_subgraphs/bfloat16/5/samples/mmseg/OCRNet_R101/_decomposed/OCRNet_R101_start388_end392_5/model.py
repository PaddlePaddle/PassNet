import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = 0.0625 * in_0;  in_0 = None
        tmp_1 = torch.nn.functional.softmax(tmp_0, dim = -1);  tmp_0 = None
        matmul = torch.matmul(tmp_1, in_1);  tmp_1 = in_1 = None
        tmp_3 = matmul.permute(0, 2, 1);  matmul = None
        return (tmp_3,)
        
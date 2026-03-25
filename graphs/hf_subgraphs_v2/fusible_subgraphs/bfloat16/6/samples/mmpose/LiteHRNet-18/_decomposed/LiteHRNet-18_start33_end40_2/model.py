import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.nn.functional.relu(in_1, inplace = False);  in_1 = None
        chunk = in_0.chunk(2, dim = 1);  in_0 = None
        tmp_2 = chunk[0]
        tmp_3 = chunk[1];  chunk = None
        chunk_1 = tmp_0.chunk(2, dim = 1);  tmp_0 = None
        tmp_5 = chunk_1[0]
        tmp_6 = chunk_1[1];  chunk_1 = None
        return (tmp_3, tmp_2, tmp_5, tmp_6)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.nn.functional.gelu(in_0, approximate = 'none');  in_0 = None
        tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False);  tmp_0 = None
        chunk = tmp_1.chunk(2, dim = -1);  tmp_1 = None
        tmp_3 = chunk[0]
        tmp_4 = chunk[1];  chunk = None
        return (tmp_3, tmp_4)
        
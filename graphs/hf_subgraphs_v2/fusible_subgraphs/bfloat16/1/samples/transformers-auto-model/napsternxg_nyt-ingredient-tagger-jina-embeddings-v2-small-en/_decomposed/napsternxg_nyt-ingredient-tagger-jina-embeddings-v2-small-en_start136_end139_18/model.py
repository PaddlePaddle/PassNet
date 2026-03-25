import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = torch.nn.functional.gelu(in_0, approximate = 'none');  in_0 = None
        tmp_1 = tmp_0 * in_1;  tmp_0 = in_1 = None
        tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False);  tmp_1 = None
        return (tmp_2,)
        
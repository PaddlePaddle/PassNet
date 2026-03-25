import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        einsum = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1);  in_2 = in_1 = None
        tmp_2 = torch.cat([in_0, einsum], dim = -1);  in_0 = einsum = None
        tmp_3 = torch.nn.functional.softmax(tmp_2, dim = -1);  tmp_2 = None
        tmp_4 = tmp_3[(Ellipsis, slice(None, 64, None))]
        return (tmp_3, tmp_4)
        
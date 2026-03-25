import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        einsum = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1);  in_4 = in_1 = None
        in_3 += einsum;  in_5 = in_3;  in_3 = einsum = None
        tmp_3 = in_5 * in_0;  in_5 = in_0 = None
        tmp_4 = tmp_3 + in_2;  tmp_3 = in_2 = None
        tmp_5 = tmp_4.contiguous();  tmp_4 = None
        return (tmp_5,)
        
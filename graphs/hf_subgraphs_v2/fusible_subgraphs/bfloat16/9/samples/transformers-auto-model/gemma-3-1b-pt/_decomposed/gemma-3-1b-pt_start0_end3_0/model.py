import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_3 = torch.nn.functional.embedding(in_0, in_2, 0, None, 2.0, False, False);  in_0 = in_2 = None
        tmp_4 = in_1.to(torch.float16);  in_1 = None
        tmp_5 = tmp_3 * tmp_4;  tmp_3 = tmp_4 = None
        return (tmp_5,)
        
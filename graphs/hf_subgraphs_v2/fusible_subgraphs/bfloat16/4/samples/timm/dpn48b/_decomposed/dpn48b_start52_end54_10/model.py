import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        tmp_4 = in_5[(slice(None, None, None), slice(128, None, None), slice(None, None, None), slice(None, None, None))];  in_5 = None
        tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001);  in_4 = in_0 = in_1 = in_3 = in_2 = None
        return (tmp_5, tmp_4)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_6 = torch.prelu(in_7, in_5);  in_7 = in_5 = None
        tmp_7 = torch.cat([tmp_6, in_6], 1);  tmp_6 = in_6 = None
        tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_0, in_1, in_3, in_2, False, 0.1, 0.001);  tmp_7 = in_0 = in_1 = in_3 = in_2 = None
        tmp_9 = torch.prelu(tmp_8, in_4);  tmp_8 = in_4 = None
        return (tmp_9,)
        
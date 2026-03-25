import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_6 = torch.prelu(in_1, w_5);  in_1 = w_5 = None
        tmp_7 = torch.cat([tmp_6, in_0], 1);  tmp_6 = in_0 = None
        tmp_8 = torch.nn.functional.batch_norm(tmp_7, w_0, w_1, w_3, w_2, False, 0.1, 0.001);  tmp_7 = w_0 = w_1 = w_3 = w_2 = None
        tmp_9 = torch.prelu(tmp_8, w_4);  tmp_8 = w_4 = None
        return (tmp_9,)
        
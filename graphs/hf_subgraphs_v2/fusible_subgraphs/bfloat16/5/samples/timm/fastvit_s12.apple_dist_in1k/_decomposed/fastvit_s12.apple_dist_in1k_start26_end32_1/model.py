import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        in_6 += in_7;  in_8 = in_6;  in_6 = in_7 = None
        tmp_6 = torch.nn.functional.batch_norm(in_5, in_1, in_2, in_4, in_3, False, 0.1, 1e-05);  in_1 = in_2 = in_4 = in_3 = None
        tmp_7 = 0 + tmp_6;  tmp_6 = None
        tmp_8 = in_8 - tmp_7;  in_8 = tmp_7 = None
        tmp_9 = tmp_8 * in_0;  tmp_8 = in_0 = None
        tmp_10 = in_5 + tmp_9;  in_5 = tmp_9 = None
        return (tmp_10,)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        tmp_2 = torch.nn.functional.dropout(in_2, 0.1, False, False);  in_2 = None
        tmp_3 = in_4 * tmp_2;  in_4 = tmp_2 = None
        tmp_4 = in_3 + tmp_3;  in_3 = tmp_3 = None
        tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 3, 1, 1, False, False, None)
        tmp_6 = tmp_5 - tmp_4;  tmp_5 = None
        tmp_7 = in_0.unsqueeze(-1);  in_0 = None
        tmp_8 = tmp_7.unsqueeze(-1);  tmp_7 = None
        tmp_9 = tmp_8 * tmp_6;  tmp_8 = tmp_6 = None
        tmp_10 = tmp_4 + tmp_9;  tmp_4 = tmp_9 = None
        tmp_11 = in_1.unsqueeze(-1);  in_1 = None
        tmp_12 = tmp_11.unsqueeze(-1);  tmp_11 = None
        return (tmp_10, tmp_12)
        
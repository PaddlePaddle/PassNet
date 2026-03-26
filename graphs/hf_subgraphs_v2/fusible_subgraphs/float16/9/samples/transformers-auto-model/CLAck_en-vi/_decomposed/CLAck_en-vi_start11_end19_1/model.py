import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = in_3.to(torch.float32);  in_3 = None
        tmp_3 = torch.tensor(1.0, dtype = torch.float32)
        tmp_4 = tmp_3 - tmp_2;  tmp_3 = tmp_2 = None
        tmp_5 = tmp_4.to(torch.bool)
        tmp_6 = tmp_4.masked_fill(tmp_5, -3.4028234663852886e+38);  tmp_4 = tmp_5 = None
        tmp_7 = torch.nn.functional.embedding(in_2, in_1, None, None, 2.0, False, False);  in_2 = in_1 = None
        tmp_8 = in_0 + tmp_7;  in_0 = tmp_7 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, p = 0.1, training = False);  tmp_8 = None
        return (tmp_6, tmp_9)
        
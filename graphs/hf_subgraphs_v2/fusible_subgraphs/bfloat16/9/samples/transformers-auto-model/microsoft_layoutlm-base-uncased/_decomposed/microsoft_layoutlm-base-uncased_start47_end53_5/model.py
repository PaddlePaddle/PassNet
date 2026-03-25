import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = in_1 * 0.125;  in_1 = None
        tmp_1 = in_0[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 11, None))];  in_0 = None
        tmp_2 = tmp_0 + tmp_1;  tmp_0 = tmp_1 = None
        tmp_3 = torch.nn.functional.softmax(tmp_2, dim = -1, dtype = torch.float32);  tmp_2 = None
        tmp_4 = tmp_3.to(torch.float32);  tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, p = 0.0, training = False);  tmp_4 = None
        return (tmp_5,)
        
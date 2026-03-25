import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        tmp_4 = torch.nn.functional.batch_norm(in_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  in_5 = in_0 = in_1 = in_3 = in_2 = None
        tmp_5 = torch.nn.functional.relu(tmp_4, inplace = True);  tmp_4 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.5, False, False);  tmp_5 = None
        tmp_7 = in_4[(slice(None, None, None), slice(None, None, None), slice(0, 1, None))];  in_4 = None
        tmp_8 = tmp_6 + tmp_7;  tmp_6 = tmp_7 = None
        return (tmp_8,)
        
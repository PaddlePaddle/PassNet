import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, in_0, in_1, in_2):
        in_1 += in_2;  in_3 = in_1;  in_1 = in_2 = None
        in_3 += in_0;  tmp_4 = in_3;  in_3 = in_0 = None
        tmp_6 = torch.nn.functional.relu(tmp_4, inplace = True);  tmp_4 = None
        tmp_7 = torch.nn.functional.batch_norm(tmp_6, w_0, w_1, w_3, w_2, False, 0.1, 1e-05);  w_0 = w_1 = w_3 = w_2 = None
        return (tmp_6, tmp_7)
        
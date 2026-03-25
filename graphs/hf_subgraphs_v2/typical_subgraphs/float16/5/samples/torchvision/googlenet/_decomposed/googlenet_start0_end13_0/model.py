import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_1 = in_0[(slice(None, None, None), 0)]
        tmp_2 = torch.unsqueeze(tmp_1, 1);  tmp_1 = None
        tmp_3 = tmp_2 * 0.458;  tmp_2 = None
        tmp_4 = tmp_3 + -0.030000000000000027;  tmp_3 = None
        tmp_5 = in_0[(slice(None, None, None), 1)]
        tmp_6 = torch.unsqueeze(tmp_5, 1);  tmp_5 = None
        tmp_7 = tmp_6 * 0.448;  tmp_6 = None
        tmp_8 = tmp_7 + -0.08799999999999997;  tmp_7 = None
        tmp_9 = in_0[(slice(None, None, None), 2)];  in_0 = None
        tmp_10 = torch.unsqueeze(tmp_9, 1);  tmp_9 = None
        tmp_11 = tmp_10 * 0.45;  tmp_10 = None
        tmp_12 = tmp_11 + -0.18799999999999994;  tmp_11 = None
        tmp_13 = torch.cat((tmp_4, tmp_8, tmp_12), 1);  tmp_4 = tmp_8 = tmp_12 = None
        return (tmp_13,)
        
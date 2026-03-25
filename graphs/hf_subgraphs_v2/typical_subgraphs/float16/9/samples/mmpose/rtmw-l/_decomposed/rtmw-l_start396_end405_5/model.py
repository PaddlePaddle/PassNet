import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1):
        tmp_2 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        tmp_3 = torch.flatten(tmp_2, 2);  tmp_2 = None
        tmp_4 = torch.functional.norm(tmp_3, dim = -1, keepdim = True)
        tmp_5 = tmp_4 * 0.14433756729740643;  tmp_4 = None
        tmp_6 = tmp_5.clamp(min = 1e-05);  tmp_5 = None
        tmp_7 = tmp_3 / tmp_6;  tmp_3 = tmp_6 = None
        tmp_8 = tmp_7 * w_0;  tmp_7 = w_0 = None
        linear = torch.nn.functional.linear(tmp_8, w_1, None);  tmp_8 = w_1 = None
        tmp_10 = torch.pixel_shuffle(in_0, 2);  in_0 = None
        return (tmp_10, linear)
        
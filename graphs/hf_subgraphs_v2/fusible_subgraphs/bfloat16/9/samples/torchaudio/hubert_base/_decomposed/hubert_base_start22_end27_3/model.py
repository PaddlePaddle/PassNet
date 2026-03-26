import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        conv1d = torch.conv1d(in_2, in_3, in_0, (1,), (64,), (1,), 16);  in_2 = in_3 = in_0 = None
        tmp_2 = conv1d[(Ellipsis, slice(None, -1, None))];  conv1d = None
        tmp_3 = torch.nn.functional.gelu(tmp_2);  tmp_2 = None
        tmp_4 = tmp_3.transpose(-2, -1);  tmp_3 = None
        tmp_5 = in_1 + tmp_4;  in_1 = tmp_4 = None
        return (tmp_5,)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_1 = in_2.mean((2, 3))
        tmp_2 = tmp_1.view(1, 1, -1);  tmp_1 = None
        conv1d = torch.conv1d(tmp_2, in_0, None, (1,), (2,), (1,), 1);  tmp_2 = in_0 = None
        tmp_4 = conv1d.sigmoid();  conv1d = None
        tmp_5 = tmp_4.view(1, -1, 1, 1);  tmp_4 = None
        tmp_6 = tmp_5.expand_as(in_2);  tmp_5 = None
        tmp_7 = in_2 * tmp_6;  in_2 = tmp_6 = None
        tmp_7 += in_1;  tmp_8 = tmp_7;  tmp_7 = in_1 = None
        return (tmp_8,)
        
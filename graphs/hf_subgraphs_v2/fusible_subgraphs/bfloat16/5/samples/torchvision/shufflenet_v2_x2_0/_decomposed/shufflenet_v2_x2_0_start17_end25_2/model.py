import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.cat((in_1, in_0), dim = 1);  in_1 = in_0 = None
        tmp_1 = tmp_0.view(32, 2, 122, 48, 48);  tmp_0 = None
        tmp_2 = torch.transpose(tmp_1, 1, 2);  tmp_1 = None
        tmp_3 = tmp_2.contiguous();  tmp_2 = None
        tmp_4 = tmp_3.view(32, 244, 48, 48);  tmp_3 = None
        chunk = tmp_4.chunk(2, dim = 1);  tmp_4 = None
        tmp_6 = chunk[0]
        tmp_7 = chunk[1];  chunk = None
        return (tmp_6, tmp_7)
        
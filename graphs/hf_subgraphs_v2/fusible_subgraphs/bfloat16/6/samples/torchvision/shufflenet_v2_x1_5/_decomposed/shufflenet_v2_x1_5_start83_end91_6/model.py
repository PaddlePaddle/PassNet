import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = torch.cat((in_0, in_1), dim = 1);  in_0 = in_1 = None
        tmp_1 = tmp_0.view(128, 2, 176, 14, 14);  tmp_0 = None
        tmp_2 = torch.transpose(tmp_1, 1, 2);  tmp_1 = None
        tmp_3 = tmp_2.contiguous();  tmp_2 = None
        tmp_4 = tmp_3.view(128, 352, 14, 14);  tmp_3 = None
        chunk = tmp_4.chunk(2, dim = 1);  tmp_4 = None
        tmp_6 = chunk[0]
        tmp_7 = chunk[1];  chunk = None
        return (tmp_6, tmp_7)
        
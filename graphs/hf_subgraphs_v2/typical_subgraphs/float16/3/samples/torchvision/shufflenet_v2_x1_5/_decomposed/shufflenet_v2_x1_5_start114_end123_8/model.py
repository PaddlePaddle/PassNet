import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_0 = torch.nn.functional.relu(in_0, inplace = True);  in_0 = None
        tmp_1 = torch.cat((in_1, tmp_0), dim = 1);  in_1 = tmp_0 = None
        tmp_2 = tmp_1.view(32, 2, 176, 14, 14);  tmp_1 = None
        tmp_3 = torch.transpose(tmp_2, 1, 2);  tmp_2 = None
        tmp_4 = tmp_3.contiguous();  tmp_3 = None
        tmp_5 = tmp_4.view(32, 352, 14, 14);  tmp_4 = None
        chunk = tmp_5.chunk(2, dim = 1);  tmp_5 = None
        tmp_7 = chunk[0]
        tmp_8 = chunk[1];  chunk = None
        return (tmp_7, tmp_8)
        
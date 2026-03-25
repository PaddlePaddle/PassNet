import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = torch.nn.functional.softmax(in_2, dim = 2);  in_2 = None
        tmp_3 = tmp_2.reshape(-1, 17, 64, 64);  tmp_2 = None
        tmp_4 = tmp_3.mul(in_0);  in_0 = None
        tmp_5 = tmp_4.reshape(32, 17, -1);  tmp_4 = None
        tmp_6 = torch.sum(tmp_5, dim = 2, keepdim = True);  tmp_5 = None
        tmp_7 = tmp_3.mul(in_1);  in_1 = None
        tmp_8 = tmp_7.reshape(32, 17, -1);  tmp_7 = None
        tmp_9 = torch.sum(tmp_8, dim = 2, keepdim = True);  tmp_8 = None
        tmp_10 = torch.cat([tmp_6, tmp_9], dim = -1);  tmp_6 = tmp_9 = None
        return (tmp_3, tmp_10)
        
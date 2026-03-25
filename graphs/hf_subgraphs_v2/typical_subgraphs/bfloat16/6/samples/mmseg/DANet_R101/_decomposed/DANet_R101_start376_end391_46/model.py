import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_1 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        tmp_2 = tmp_1.view(24, 512, -1)
        tmp_3 = tmp_1.view(24, 512, -1)
        tmp_4 = tmp_3.permute(0, 2, 1);  tmp_3 = None
        bmm = torch.bmm(tmp_2, tmp_4);  tmp_2 = tmp_4 = None
        max_1 = torch.max(bmm, -1, keepdim = True)
        tmp_7 = max_1[0];  max_1 = None
        tmp_8 = tmp_7.expand_as(bmm);  tmp_7 = None
        tmp_9 = tmp_8 - bmm;  tmp_8 = bmm = None
        tmp_10 = torch.nn.functional.softmax(tmp_9, dim = -1);  tmp_9 = None
        tmp_11 = tmp_1.view(24, 512, -1)
        bmm_1 = torch.bmm(tmp_10, tmp_11);  tmp_10 = tmp_11 = None
        tmp_13 = bmm_1.view(24, 512, 64, 64);  bmm_1 = None
        tmp_14 = tmp_13 * in_0;  tmp_13 = in_0 = None
        tmp_15 = tmp_14 + tmp_1;  tmp_14 = tmp_1 = None
        return (tmp_15,)
        
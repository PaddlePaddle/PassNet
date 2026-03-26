import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        bmm = torch.bmm(in_0, in_1);  in_0 = in_1 = None
        tmp_1 = torch.nn.functional.softmax(bmm, dim = -1);  bmm = None
        tmp_2 = torch.nn.functional.dropout(tmp_1, p = 0.0, training = False);  tmp_1 = None
        bmm_1 = torch.bmm(tmp_2, in_2);  tmp_2 = in_2 = None
        tmp_4 = bmm_1.view(1, 8, 1, 32);  bmm_1 = None
        tmp_5 = tmp_4.transpose(1, 2);  tmp_4 = None
        tmp_6 = tmp_5.reshape(1, 1, 256);  tmp_5 = None
        return (tmp_6,)
        
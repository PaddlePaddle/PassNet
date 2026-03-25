import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_1 = in_0[in_2];  in_0 = in_2 = None
        tmp_2 = tmp_1.view(49, 49, -1);  tmp_1 = None
        tmp_3 = tmp_2.permute(2, 0, 1);  tmp_2 = None
        tmp_4 = tmp_3.contiguous();  tmp_3 = None
        tmp_5 = tmp_4.unsqueeze(0);  tmp_4 = None
        tmp_6 = in_1 + tmp_5;  in_1 = tmp_5 = None
        tmp_7 = torch.nn.functional.softmax(tmp_6, -1, _stacklevel = 5);  tmp_6 = None
        tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False);  tmp_7 = None
        return (tmp_8,)
        
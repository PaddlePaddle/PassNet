import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_2 = in_2 / 5.656854249492381;  in_2 = None
        tmp_3 = in_0.view(-1);  in_0 = None
        tmp_4 = in_1[tmp_3];  in_1 = tmp_3 = None
        tmp_5 = tmp_4.view(49, 49, -1);  tmp_4 = None
        tmp_6 = tmp_5.permute(2, 0, 1);  tmp_5 = None
        tmp_7 = tmp_6.contiguous();  tmp_6 = None
        tmp_8 = tmp_7.unsqueeze(0);  tmp_7 = None
        tmp_9 = tmp_2 + tmp_8;  tmp_2 = tmp_8 = None
        tmp_10 = tmp_9.view(1, 16, 12, 49, 49);  tmp_9 = None
        tmp_11 = in_3.unsqueeze(1);  in_3 = None
        tmp_12 = tmp_11.unsqueeze(0);  tmp_11 = None
        tmp_13 = tmp_10 + tmp_12;  tmp_10 = tmp_12 = None
        tmp_14 = tmp_13.view(-1, 12, 49, 49);  tmp_13 = None
        tmp_15 = torch.nn.functional.softmax(tmp_14, dim = -1);  tmp_14 = None
        tmp_16 = torch.nn.functional.dropout(tmp_15, 0.0, False, False);  tmp_15 = None
        return (tmp_16,)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_3, in_1, None);  in_3 = in_1 = None
        tmp_3 = linear.view(-1, 2);  linear = None
        tmp_4 = in_0.view(-1);  in_0 = None
        tmp_5 = tmp_3[tmp_4];  tmp_3 = tmp_4 = None
        tmp_6 = tmp_5.view(4, 4, -1);  tmp_5 = None
        tmp_7 = tmp_6.permute(2, 0, 1);  tmp_6 = None
        tmp_8 = tmp_7.contiguous();  tmp_7 = None
        tmp_9 = torch.sigmoid(tmp_8);  tmp_8 = None
        tmp_10 = 16 * tmp_9;  tmp_9 = None
        tmp_11 = tmp_10.unsqueeze(0);  tmp_10 = None
        tmp_12 = in_2 + tmp_11;  in_2 = tmp_11 = None
        return (tmp_12,)
        
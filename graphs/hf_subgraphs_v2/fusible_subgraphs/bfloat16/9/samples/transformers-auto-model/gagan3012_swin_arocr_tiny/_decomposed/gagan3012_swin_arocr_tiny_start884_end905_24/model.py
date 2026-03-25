import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        linear = torch.nn.functional.linear(in_4, in_1, None);  in_4 = in_1 = None
        tmp_3 = linear.view(-1, 24);  linear = None
        tmp_4 = in_0.view(-1);  in_0 = None
        tmp_5 = tmp_3[tmp_4];  tmp_3 = tmp_4 = None
        tmp_6 = tmp_5.view(64, 64, -1);  tmp_5 = None
        tmp_7 = tmp_6.permute(2, 0, 1);  tmp_6 = None
        tmp_8 = tmp_7.contiguous();  tmp_7 = None
        tmp_9 = torch.sigmoid(tmp_8);  tmp_8 = None
        tmp_10 = 16 * tmp_9;  tmp_9 = None
        tmp_11 = tmp_10.unsqueeze(0);  tmp_10 = None
        tmp_12 = in_2 + tmp_11;  in_2 = tmp_11 = None
        tmp_13 = tmp_12.view(1, 16, 24, 64, 64);  tmp_12 = None
        tmp_14 = in_3.unsqueeze(1)
        tmp_15 = tmp_14.unsqueeze(0);  tmp_14 = None
        tmp_16 = tmp_13 + tmp_15;  tmp_13 = tmp_15 = None
        tmp_17 = in_3.unsqueeze(1);  in_3 = None
        tmp_18 = tmp_17.unsqueeze(0);  tmp_17 = None
        tmp_19 = tmp_16 + tmp_18;  tmp_16 = tmp_18 = None
        tmp_20 = tmp_19.view(-1, 24, 64, 64);  tmp_19 = None
        tmp_21 = torch.nn.functional.softmax(tmp_20, dim = -1);  tmp_20 = None
        tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False);  tmp_21 = None
        return (tmp_22,)
        
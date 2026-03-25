import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13):
        tmp_9 = torch.nn.functional.gelu(in_5);  in_5 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.1, False, False);  tmp_9 = None
        linear = torch.nn.functional.linear(tmp_10, w_3, w_2);  tmp_10 = w_3 = w_2 = None
        tmp_12 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_13 = in_4 + tmp_12;  in_4 = tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (768,), w_5, w_4, 1e-05);  tmp_13 = w_5 = w_4 = None
        tmp_15 = torch.stack((in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_0, in_1, in_2, in_3, tmp_14), dim = 1);  in_6 = in_7 = in_8 = in_9 = in_10 = in_11 = in_12 = in_13 = in_0 = in_1 = in_2 = in_3 = None
        tmp_16 = torch.nn.functional.softmax(w_8, dim = -1);  w_8 = None
        tmp_17 = tmp_16.view(-1, 1, 1);  tmp_16 = None
        tmp_18 = tmp_15 * tmp_17;  tmp_15 = tmp_17 = None
        tmp_19 = tmp_18.sum(dim = 1);  tmp_18 = None
        linear_1 = torch.nn.functional.linear(tmp_19, w_7, w_6);  tmp_19 = w_7 = w_6 = None
        tmp_21 = linear_1.mean(dim = 1);  linear_1 = None
        linear_2 = torch.nn.functional.linear(tmp_21, w_1, w_0);  tmp_21 = w_1 = w_0 = None
        return (tmp_14, linear_2)
        
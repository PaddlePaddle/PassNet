import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
        tmp_10 = torch.nn.functional.gelu(in_11, approximate = 'none');  in_11 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False);  tmp_10 = None
        chunk = tmp_11.chunk(2, dim = -1);  tmp_11 = None
        tmp_13 = chunk[0]
        tmp_14 = chunk[1];  chunk = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (768,), in_3, in_2, 1e-05);  tmp_14 = in_3 = in_2 = None
        tmp_16 = tmp_15.transpose(-1, -2);  tmp_15 = None
        linear = torch.nn.functional.linear(tmp_16, in_5, in_4);  tmp_16 = in_5 = in_4 = None
        tmp_18 = linear.transpose(-1, -2);  linear = None
        tmp_19 = tmp_13 * tmp_18;  tmp_13 = tmp_18 = None
        linear_1 = torch.nn.functional.linear(tmp_19, in_1, in_0);  tmp_19 = in_1 = in_0 = None
        tmp_21 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_22 = in_10 + tmp_21;  in_10 = tmp_21 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (256,), in_9, in_8, 1e-06);  tmp_22 = in_9 = in_8 = None
        tmp_24 = tmp_23.mean(dim = 1);  tmp_23 = None
        tmp_25 = torch.nn.functional.dropout(tmp_24, 0.0, False, False);  tmp_24 = None
        linear_2 = torch.nn.functional.linear(tmp_25, in_7, in_6);  tmp_25 = in_7 = in_6 = None
        return (linear_2,)
        
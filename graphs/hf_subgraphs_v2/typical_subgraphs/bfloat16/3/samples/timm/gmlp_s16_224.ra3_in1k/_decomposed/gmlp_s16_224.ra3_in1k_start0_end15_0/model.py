import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor):
        conv2d = torch.conv2d(in_10, in_9, in_8, (16, 16), (0, 0), (1, 1), 1);  in_10 = in_9 = in_8 = None
        tmp_12 = conv2d.flatten(2);  conv2d = None
        tmp_13 = tmp_12.transpose(1, 2);  tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (256,), in_7, in_6, 1e-06);  in_7 = in_6 = None
        linear = torch.nn.functional.linear(tmp_14, in_1, in_0);  tmp_14 = in_1 = in_0 = None
        tmp_16 = torch.nn.functional.gelu(linear, approximate = 'none');  linear = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.0, False, False);  tmp_16 = None
        chunk = tmp_17.chunk(2, dim = -1);  tmp_17 = None
        tmp_19 = chunk[0]
        tmp_20 = chunk[1];  chunk = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (768,), in_3, in_2, 1e-05);  tmp_20 = in_3 = in_2 = None
        tmp_22 = tmp_21.transpose(-1, -2);  tmp_21 = None
        linear_1 = torch.nn.functional.linear(tmp_22, in_5, in_4);  tmp_22 = in_5 = in_4 = None
        tmp_24 = linear_1.transpose(-1, -2);  linear_1 = None
        tmp_25 = tmp_19 * tmp_24;  tmp_19 = tmp_24 = None
        return (tmp_13, tmp_25)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, in_0 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_9, w_8, (16, 16), (0, 0), (1, 1), 1);  in_0 = w_9 = w_8 = None
        tmp_12 = conv2d.flatten(2);  conv2d = None
        tmp_13 = tmp_12.transpose(1, 2);  tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (256,), w_7, w_6, 1e-06);  w_7 = w_6 = None
        linear = torch.nn.functional.linear(tmp_14, w_1, w_0);  tmp_14 = w_1 = w_0 = None
        tmp_16 = torch.nn.functional.gelu(linear, approximate = 'none');  linear = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.0, False, False);  tmp_16 = None
        chunk = tmp_17.chunk(2, dim = -1);  tmp_17 = None
        tmp_19 = chunk[0]
        tmp_20 = chunk[1];  chunk = None
        tmp_21 = torch.nn.functional.layer_norm(tmp_20, (768,), w_3, w_2, 1e-05);  tmp_20 = w_3 = w_2 = None
        tmp_22 = tmp_21.transpose(-1, -2);  tmp_21 = None
        linear_1 = torch.nn.functional.linear(tmp_22, w_5, w_4);  tmp_22 = w_5 = w_4 = None
        tmp_24 = linear_1.transpose(-1, -2);  linear_1 = None
        tmp_25 = tmp_19 * tmp_24;  tmp_19 = tmp_24 = None
        return (tmp_13, tmp_25)
        
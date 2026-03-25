import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor):
        conv2d = torch.conv2d(in_9, in_6, None, (16, 16), (0, 0), (1, 1), 1);  in_9 = in_6 = None
        tmp_11 = conv2d.flatten(2);  conv2d = None
        tmp_12 = tmp_11.transpose(1, 2);  tmp_11 = None
        tmp_13 = in_7.expand(1, -1, -1);  in_7 = None
        tmp_14 = torch.cat([tmp_13, tmp_12], dim = 1);  tmp_13 = tmp_12 = None
        tmp_15 = tmp_14 + in_8;  tmp_14 = in_8 = None
        tmp_16 = torch.nn.functional.dropout(tmp_15, 0.0, False, False);  tmp_15 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (256,), in_5, in_4, 1e-05);  tmp_16 = in_5 = in_4 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (256,), in_3, in_2, 1e-05);  in_3 = in_2 = None
        linear = torch.nn.functional.linear(tmp_18, in_1, in_0);  tmp_18 = in_1 = in_0 = None
        return (linear, tmp_17)
        
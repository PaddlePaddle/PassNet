import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, in_0 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_5, w_4, (14, 14), (0, 0), (1, 1), 1);  in_0 = w_5 = w_4 = None
        tmp_10 = conv2d.flatten(2);  conv2d = None
        tmp_11 = tmp_10.transpose(1, 2);  tmp_10 = None
        tmp_12 = w_6.expand(1, -1, -1);  w_6 = None
        tmp_13 = torch.cat([tmp_12, tmp_11], dim = 1);  tmp_12 = tmp_11 = None
        tmp_14 = tmp_13 + w_7;  tmp_13 = w_7 = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, 0.0, False, False);  tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (1280,), w_3, w_2, 1e-06);  w_3 = w_2 = None
        linear = torch.nn.functional.linear(tmp_16, w_1, w_0);  tmp_16 = w_1 = w_0 = None
        return (linear, tmp_15)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, in_0 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_5, w_4, (16, 16), (0, 0), (1, 1), 1);  in_0 = w_5 = w_4 = None
        tmp_9 = conv2d.flatten(2);  conv2d = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        tmp_11 = tmp_10 + w_6;  tmp_10 = w_6 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False);  tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), w_3, w_2, 1e-06);  w_3 = w_2 = None
        linear = torch.nn.functional.linear(tmp_13, w_1, w_0);  tmp_13 = w_1 = w_0 = None
        return (linear, tmp_12)
        
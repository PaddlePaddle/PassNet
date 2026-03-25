import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_1, w_0, (16, 16), (0, 0), (1, 1), 1);  in_0 = w_1 = w_0 = None
        tmp_9 = conv2d.flatten(2);  conv2d = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        tmp_11 = w_2.expand(1, -1, -1);  w_2 = None
        tmp_12 = w_3.expand(1, -1, -1);  w_3 = None
        tmp_13 = torch.cat((tmp_11, tmp_12, tmp_10), dim = 1);  tmp_11 = tmp_12 = tmp_10 = None
        tmp_14 = tmp_13 + w_4;  tmp_13 = w_4 = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, 0.0, False, False);  tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (192,), w_6, w_5, 1e-12);  w_6 = w_5 = None
        return (tmp_15, tmp_16)
        
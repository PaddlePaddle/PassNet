import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_1, w_0, (8, 8), (0, 0), (1, 1), 1);  in_0 = w_1 = w_0 = None
        tmp_8 = conv2d.flatten(2);  conv2d = None
        tmp_9 = tmp_8.transpose(1, 2);  tmp_8 = None
        tmp_10 = w_2.expand(1, -1, -1);  w_2 = None
        tmp_11 = torch.cat((tmp_10, tmp_9), dim = 1);  tmp_10 = tmp_9 = None
        tmp_12 = tmp_11 + w_3;  tmp_11 = w_3 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False);  tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(tmp_13, (768,), w_5, w_4, 1e-12);  w_5 = w_4 = None
        return (tmp_13, tmp_14)
        
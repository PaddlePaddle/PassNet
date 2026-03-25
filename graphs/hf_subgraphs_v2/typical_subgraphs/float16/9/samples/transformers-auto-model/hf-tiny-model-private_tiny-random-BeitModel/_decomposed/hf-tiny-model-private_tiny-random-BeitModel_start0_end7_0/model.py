import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor):
        conv2d = torch.conv2d(in_0, w_1, w_0, (2, 2), (0, 0), (1, 1), 1);  in_0 = w_1 = w_0 = None
        tmp_7 = conv2d.flatten(2);  conv2d = None
        tmp_8 = tmp_7.transpose(1, 2);  tmp_7 = None
        tmp_9 = w_2.expand(1, -1, -1);  w_2 = None
        tmp_10 = torch.cat((tmp_9, tmp_8), dim = 1);  tmp_9 = tmp_8 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.1, False, False);  tmp_10 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (32,), w_4, w_3, 1e-12);  w_4 = w_3 = None
        return (tmp_11, tmp_12)
        
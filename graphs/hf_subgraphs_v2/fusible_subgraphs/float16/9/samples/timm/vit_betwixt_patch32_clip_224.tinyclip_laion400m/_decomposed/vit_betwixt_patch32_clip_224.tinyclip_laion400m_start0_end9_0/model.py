import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor):
        conv2d = torch.conv2d(in_7, in_4, None, (32, 32), (0, 0), (1, 1), 1);  in_7 = in_4 = None
        tmp_9 = conv2d.flatten(2);  conv2d = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        tmp_11 = in_5.expand(1, -1, -1);  in_5 = None
        tmp_12 = torch.cat([tmp_11, tmp_10], dim = 1);  tmp_11 = tmp_10 = None
        tmp_13 = tmp_12 + in_6;  tmp_12 = in_6 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False);  tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (640,), in_3, in_2, 1e-05);  tmp_14 = in_3 = in_2 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (640,), in_1, in_0, 1e-05);  in_1 = in_0 = None
        return (tmp_15, tmp_16)
        
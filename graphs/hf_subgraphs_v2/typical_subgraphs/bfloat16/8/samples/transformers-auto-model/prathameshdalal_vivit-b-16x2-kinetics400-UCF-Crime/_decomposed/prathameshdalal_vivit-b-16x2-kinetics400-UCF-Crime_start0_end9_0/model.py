import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        tmp_7 = in_0.permute(0, 2, 1, 3, 4);  in_0 = None
        conv3d = torch.conv3d(tmp_7, in_2, in_1, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1);  tmp_7 = in_2 = in_1 = None
        tmp_9 = conv3d.flatten(2);  conv3d = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        tmp_11 = in_3.tile([1, 1, 1]);  in_3 = None
        tmp_12 = torch.cat((tmp_11, tmp_10), dim = 1);  tmp_11 = tmp_10 = None
        tmp_13 = tmp_12 + in_4;  tmp_12 = in_4 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False);  tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (768,), in_6, in_5, 1e-06);  in_6 = in_5 = None
        return (tmp_14, tmp_15)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor):
        tmp_7 = in_0.permute(0, 2, 1, 3, 4);  in_0 = None
        conv3d = torch.conv3d(tmp_7, w_1, w_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1);  tmp_7 = w_1 = w_0 = None
        tmp_9 = conv3d.flatten(2);  conv3d = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        tmp_11 = w_2.tile([1, 1, 1]);  w_2 = None
        tmp_12 = torch.cat((tmp_11, tmp_10), dim = 1);  tmp_11 = tmp_10 = None
        tmp_13 = tmp_12 + w_3;  tmp_12 = w_3 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False);  tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (768,), w_5, w_4, 1e-06);  w_5 = w_4 = None
        return (tmp_14, tmp_15)
        
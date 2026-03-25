import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor):
        tmp_8 = in_0.unsqueeze(1);  in_0 = None
        tmp_9 = tmp_8.transpose(2, 3);  tmp_8 = None
        conv2d = torch.conv2d(tmp_9, w_1, w_0, (14, 14), (0, 0), (1, 1), 1);  tmp_9 = w_1 = w_0 = None
        tmp_11 = conv2d.flatten(2);  conv2d = None
        tmp_12 = tmp_11.transpose(1, 2);  tmp_11 = None
        tmp_13 = w_2.expand(1, -1, -1);  w_2 = None
        tmp_14 = w_3.expand(1, -1, -1);  w_3 = None
        tmp_15 = torch.cat((tmp_13, tmp_14, tmp_12), dim = 1);  tmp_13 = tmp_14 = tmp_12 = None
        tmp_16 = tmp_15 + w_4;  tmp_15 = w_4 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.0, False, False);  tmp_16 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (768,), w_6, w_5, 1e-12);  w_6 = w_5 = None
        return (tmp_17, tmp_18)
        
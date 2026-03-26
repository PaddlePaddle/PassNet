import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor):
        conv2d = torch.conv2d(in_8, in_1, None, (14, 14), (0, 0), (1, 1), 1);  in_8 = in_1 = None
        tmp_9 = conv2d.flatten(2);  conv2d = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        tmp_11 = in_3.expand(1, 1, -1);  in_3 = None
        tmp_12 = torch.cat([tmp_11, tmp_10], dim = 1);  tmp_11 = tmp_10 = None
        tmp_13 = torch.nn.functional.embedding(in_0, in_2, None, None, 2.0, False, False);  in_0 = in_2 = None
        tmp_14 = tmp_12 + tmp_13;  tmp_12 = tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (1280,), in_7, in_6, 1e-05);  tmp_14 = in_7 = in_6 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (1280,), in_5, in_4, 1e-05);  in_5 = in_4 = None
        return (tmp_15, tmp_16)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        tmp_4 = in_5.view(32, -1, 2, 32);  in_5 = None
        tmp_5 = tmp_4.transpose(1, 2);  tmp_4 = None
        tmp_6 = in_4.permute(0, 2, 1);  in_4 = None
        tmp_7 = tmp_6.reshape(32, 64, 64, 64);  tmp_6 = None
        conv2d = torch.conv2d(tmp_7, in_3, in_2, (4, 4), (0, 0), (1, 1), 1);  tmp_7 = in_3 = in_2 = None
        tmp_9 = conv2d.reshape(32, 64, -1);  conv2d = None
        tmp_10 = tmp_9.permute(0, 2, 1);  tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (64,), in_1, in_0, 1e-05);  tmp_10 = in_1 = in_0 = None
        return (tmp_11, tmp_5)
        
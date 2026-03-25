import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, in_0, in_1):
        tmp_4 = in_1.view(1, -1, 5, 32);  in_1 = None
        tmp_5 = tmp_4.transpose(1, 2);  tmp_4 = None
        tmp_6 = in_0.permute(0, 2, 1);  in_0 = None
        tmp_7 = tmp_6.reshape(1, 160, 32, 32);  tmp_6 = None
        conv2d = torch.conv2d(tmp_7, w_3, w_2, (2, 2), (0, 0), (1, 1), 1);  tmp_7 = w_3 = w_2 = None
        tmp_9 = conv2d.reshape(1, 160, -1);  conv2d = None
        tmp_10 = tmp_9.permute(0, 2, 1);  tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (160,), w_1, w_0, 1e-05);  tmp_10 = w_1 = w_0 = None
        return (tmp_11, tmp_5)
        
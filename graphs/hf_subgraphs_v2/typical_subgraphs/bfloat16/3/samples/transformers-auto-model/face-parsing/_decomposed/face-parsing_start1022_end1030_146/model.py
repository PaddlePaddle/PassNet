import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        tmp_4 = in_5.view(8, -1, 5, 64);  in_5 = None
        tmp_5 = tmp_4.transpose(1, 2);  tmp_4 = None
        tmp_6 = in_4.permute(0, 2, 1);  in_4 = None
        tmp_7 = tmp_6.reshape(8, 320, 32, 32);  tmp_6 = None
        to = tmp_7.to(torch.bfloat16);  tmp_7 = None
        conv2d = torch.conv2d(to, in_3, in_2, (2, 2), (0, 0), (1, 1), 1);  to = in_3 = in_2 = None
        tmp_9 = conv2d.reshape(8, 320, -1);  conv2d = None
        tmp_10 = tmp_9.permute(0, 2, 1);  tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (320,), in_1, in_0, 1e-05);  tmp_10 = in_1 = in_0 = None
        return (tmp_11, tmp_5)
        
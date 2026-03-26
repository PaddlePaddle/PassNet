import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_2 = torch.nn.functional.gelu(in_0, approximate = 'none');  in_0 = None
        conv2d = torch.conv2d(tmp_2, w_0, None, (1, 1), (0, 0), (1, 1), 1);  tmp_2 = w_0 = None
        tmp_4 = in_1 + conv2d;  in_1 = conv2d = None
        split = torch.functional.split(tmp_4, [128, 384], dim = 1)
        tmp_6 = split[0]
        tmp_7 = split[1];  split = None
        conv2d_1 = torch.conv2d(tmp_6, w_1, None, (1, 1), (1, 1), (1, 1), 1);  tmp_6 = w_1 = None
        tmp_9 = torch.cat((conv2d_1, tmp_7), 1);  conv2d_1 = tmp_7 = None
        return (tmp_4, tmp_9)
        
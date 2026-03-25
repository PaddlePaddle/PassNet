import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        conv2d = torch.conv2d(input = in_9, weight = in_4, groups = 512);  in_9 = in_4 = None
        tmp_5 = conv2d.view(1, 512, 64, 64);  conv2d = None
        tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_5 = in_0 = in_1 = in_3 = in_2 = None
        tmp_7 = torch.nn.functional.relu(tmp_6, inplace = False);  tmp_6 = None
        tmp_8 = torch.cat([in_5, in_7, in_8, in_6, tmp_7], dim = 1);  in_5 = in_7 = in_8 = in_6 = tmp_7 = None
        return (tmp_8,)
        
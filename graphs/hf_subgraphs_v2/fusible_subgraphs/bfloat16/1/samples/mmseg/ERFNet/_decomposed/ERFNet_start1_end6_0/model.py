import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor):
        tmp_5 = torch.nn.functional.max_pool2d(in_0, 2, 2, 0, 1, ceil_mode = False, return_indices = False);  in_0 = None
        tmp_6 = torch.nn.functional.interpolate(tmp_5, (256, 256), None, 'bilinear', False);  tmp_5 = None
        tmp_7 = torch.cat([in_5, tmp_6], 1);  in_5 = tmp_6 = None
        tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_1, in_2, in_4, in_3, False, 0.1, 0.001);  tmp_7 = in_1 = in_2 = in_4 = in_3 = None
        tmp_9 = torch.nn.functional.relu(tmp_8, inplace = False);  tmp_8 = None
        return (tmp_9,)
        
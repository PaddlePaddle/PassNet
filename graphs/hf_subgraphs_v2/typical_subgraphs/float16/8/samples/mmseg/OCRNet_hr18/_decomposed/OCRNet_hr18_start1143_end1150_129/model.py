import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        tmp_2 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        conv2d = torch.conv2d(tmp_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_2 = in_1 = in_0 = None
        tmp_4 = torch.nn.functional.interpolate(in_3, (128, 128), None, 'bilinear', False);  in_3 = None
        tmp_5 = torch.nn.functional.interpolate(in_4, (128, 128), None, 'bilinear', False);  in_4 = None
        tmp_6 = torch.nn.functional.interpolate(in_5, (128, 128), None, 'bilinear', False);  in_5 = None
        tmp_7 = torch.nn.functional.interpolate(in_6, (128, 128), None, 'bilinear', False);  in_6 = None
        tmp_8 = torch.cat([tmp_4, tmp_5, tmp_6, tmp_7], dim = 1);  tmp_4 = tmp_5 = tmp_6 = tmp_7 = None
        return (conv2d, tmp_8)
        
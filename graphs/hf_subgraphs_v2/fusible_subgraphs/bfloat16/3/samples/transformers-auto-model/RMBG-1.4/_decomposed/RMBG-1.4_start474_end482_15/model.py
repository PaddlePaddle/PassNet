import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor):
        conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), 1);  in_2 = in_1 = in_0 = None
        tmp_3 = torch.nn.functional.interpolate(conv2d, size = (640, 640), mode = 'bilinear');  conv2d = None
        tmp_4 = torch.nn.functional.sigmoid(in_3);  in_3 = None
        tmp_5 = torch.nn.functional.sigmoid(in_4);  in_4 = None
        tmp_6 = torch.nn.functional.sigmoid(in_5);  in_5 = None
        tmp_7 = torch.nn.functional.sigmoid(in_6);  in_6 = None
        tmp_8 = torch.nn.functional.sigmoid(in_7);  in_7 = None
        tmp_9 = torch.nn.functional.sigmoid(tmp_3);  tmp_3 = None
        return (tmp_4, tmp_5, tmp_6, tmp_7, tmp_8, tmp_9)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor):
        conv2d = torch.conv2d(in_10, in_8, in_7, (1, 1), (0, 0), (1, 1), 1);  in_10 = in_8 = in_7 = None
        tmp_11 = torch.nn.functional.interpolate(conv2d, size = (512, 512), mode = 'bilinear', align_corners = False);  conv2d = None
        conv2d_1 = torch.conv2d(in_9, in_6, None, (1, 1), (1, 1), (1, 1), 1);  in_9 = in_6 = None
        tmp_13 = torch.nn.functional.batch_norm(conv2d_1, in_2, in_3, in_5, in_4, False, 0.1, 1e-05);  conv2d_1 = in_2 = in_3 = in_5 = in_4 = None
        tmp_14 = torch.nn.functional.relu(tmp_13, inplace = False);  tmp_13 = None
        to = tmp_14.to(torch.float16);  tmp_14 = None
        conv2d_2 = torch.conv2d(to, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  to = in_1 = in_0 = None
        tmp_16 = torch.nn.functional.interpolate(conv2d_2, size = (512, 512), mode = 'bilinear', align_corners = False);  conv2d_2 = tmp_16 = None
        return (tmp_11,)
        
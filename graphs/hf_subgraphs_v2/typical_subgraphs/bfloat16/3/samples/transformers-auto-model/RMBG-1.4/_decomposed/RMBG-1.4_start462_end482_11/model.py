import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18):
        tmp_12 = torch.nn.functional.relu(in_12, inplace = True);  in_12 = None
        tmp_13 = tmp_12 + in_18;  tmp_12 = in_18 = None
        to = tmp_13.to(torch.bfloat16)
        conv2d = torch.conv2d(to, in_1, in_0, (1, 1), (1, 1), (1, 1), 1);  to = in_1 = in_0 = None
        tmp_15 = torch.nn.functional.interpolate(conv2d, size = (640, 640), mode = 'bilinear');  conv2d = None
        conv2d_1 = torch.conv2d(in_13, in_3, in_2, (1, 1), (1, 1), (1, 1), 1);  in_13 = in_3 = in_2 = None
        tmp_17 = torch.nn.functional.interpolate(conv2d_1, size = (640, 640), mode = 'bilinear');  conv2d_1 = None
        conv2d_2 = torch.conv2d(in_14, in_5, in_4, (1, 1), (1, 1), (1, 1), 1);  in_14 = in_5 = in_4 = None
        tmp_19 = torch.nn.functional.interpolate(conv2d_2, size = (640, 640), mode = 'bilinear');  conv2d_2 = None
        conv2d_3 = torch.conv2d(in_15, in_7, in_6, (1, 1), (1, 1), (1, 1), 1);  in_15 = in_7 = in_6 = None
        tmp_21 = torch.nn.functional.interpolate(conv2d_3, size = (640, 640), mode = 'bilinear');  conv2d_3 = None
        conv2d_4 = torch.conv2d(in_16, in_9, in_8, (1, 1), (1, 1), (1, 1), 1);  in_16 = in_9 = in_8 = None
        tmp_23 = torch.nn.functional.interpolate(conv2d_4, size = (640, 640), mode = 'bilinear');  conv2d_4 = None
        conv2d_5 = torch.conv2d(in_17, in_11, in_10, (1, 1), (1, 1), (1, 1), 1);  in_17 = in_11 = in_10 = None
        tmp_25 = torch.nn.functional.interpolate(conv2d_5, size = (640, 640), mode = 'bilinear');  conv2d_5 = None
        tmp_26 = torch.nn.functional.sigmoid(tmp_15);  tmp_15 = None
        tmp_27 = torch.nn.functional.sigmoid(tmp_17);  tmp_17 = None
        tmp_28 = torch.nn.functional.sigmoid(tmp_19);  tmp_19 = None
        tmp_29 = torch.nn.functional.sigmoid(tmp_21);  tmp_21 = None
        tmp_30 = torch.nn.functional.sigmoid(tmp_23);  tmp_23 = None
        tmp_31 = torch.nn.functional.sigmoid(tmp_25);  tmp_25 = None
        return (tmp_13, tmp_26, tmp_27, tmp_28, tmp_29, tmp_30, tmp_31)
        
import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor):
        tmp_21 = torch.nn.functional.interpolate(in_20, size = (408, 408), mode = 'bicubic', align_corners = False)
        conv2d = torch.conv2d(tmp_21, in_5, in_4, (4, 4), (3, 3), (1, 1), 1);  tmp_21 = in_5 = in_4 = None
        tmp_23 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_23, in_7, in_6, (3, 3), (0, 0), (1, 1), 1);  tmp_23 = in_7 = in_6 = None
        tmp_25 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        conv2d_2 = torch.conv2d(tmp_25, in_9, in_8, (1, 1), (1, 1), (1, 1), 1);  tmp_25 = in_9 = in_8 = None
        tmp_27 = conv2d_2.flatten(2);  conv2d_2 = None
        tmp_28 = tmp_27.transpose(1, 2);  tmp_27 = None
        tmp_29 = in_16.expand(1, -1, -1);  in_16 = None
        tmp_30 = torch.cat((tmp_29, tmp_28), dim = 1);  tmp_29 = tmp_28 = None
        tmp_31 = tmp_30 + in_18;  tmp_30 = in_18 = None
        tmp_32 = torch.nn.functional.dropout(tmp_31, 0.0, False, False);  tmp_31 = None
        tmp_33 = torch.nn.functional.interpolate(in_20, size = (384, 384), mode = 'bicubic', align_corners = False);  in_20 = None
        conv2d_3 = torch.conv2d(tmp_33, in_11, in_10, (4, 4), (3, 3), (1, 1), 1);  tmp_33 = in_11 = in_10 = None
        tmp_35 = torch.nn.functional.relu(conv2d_3, inplace = True);  conv2d_3 = None
        conv2d_4 = torch.conv2d(tmp_35, in_13, in_12, (2, 2), (1, 1), (1, 1), 1);  tmp_35 = in_13 = in_12 = None
        tmp_37 = torch.nn.functional.relu(conv2d_4, inplace = True);  conv2d_4 = None
        conv2d_5 = torch.conv2d(tmp_37, in_15, in_14, (2, 2), (1, 1), (1, 1), 1);  tmp_37 = in_15 = in_14 = None
        tmp_39 = conv2d_5.flatten(2);  conv2d_5 = None
        tmp_40 = tmp_39.transpose(1, 2);  tmp_39 = None
        tmp_41 = in_17.expand(1, -1, -1);  in_17 = None
        tmp_42 = torch.cat((tmp_41, tmp_40), dim = 1);  tmp_41 = tmp_40 = None
        tmp_43 = tmp_42 + in_19;  tmp_42 = in_19 = None
        tmp_44 = torch.nn.functional.dropout(tmp_43, 0.0, False, False);  tmp_43 = None
        tmp_45 = torch.nn.functional.layer_norm(tmp_32, (192,), in_3, in_2, 1e-06);  in_3 = in_2 = None
        linear = torch.nn.functional.linear(tmp_45, in_1, in_0);  tmp_45 = in_1 = in_0 = None
        return (linear, tmp_32, tmp_44)
        
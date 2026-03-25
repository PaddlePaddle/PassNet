import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor, w_14 : torch.Tensor, w_15 : torch.Tensor, w_16 : torch.Tensor, w_17 : torch.Tensor, w_18 : torch.Tensor, w_19 : torch.Tensor, in_0 : torch.Tensor):
        tmp_21 = torch.nn.functional.interpolate(in_0, size = (240, 240), mode = 'bicubic', align_corners = False)
        conv2d = torch.conv2d(tmp_21, w_5, w_4, (4, 4), (3, 3), (1, 1), 1);  tmp_21 = w_5 = w_4 = None
        tmp_23 = torch.nn.functional.relu(conv2d, inplace = True);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_23, w_7, w_6, (3, 3), (0, 0), (1, 1), 1);  tmp_23 = w_7 = w_6 = None
        tmp_25 = torch.nn.functional.relu(conv2d_1, inplace = True);  conv2d_1 = None
        conv2d_2 = torch.conv2d(tmp_25, w_9, w_8, (1, 1), (1, 1), (1, 1), 1);  tmp_25 = w_9 = w_8 = None
        tmp_27 = conv2d_2.flatten(2);  conv2d_2 = None
        tmp_28 = tmp_27.transpose(1, 2);  tmp_27 = None
        tmp_29 = w_16.expand(1, -1, -1);  w_16 = None
        tmp_30 = torch.cat((tmp_29, tmp_28), dim = 1);  tmp_29 = tmp_28 = None
        tmp_31 = tmp_30 + w_18;  tmp_30 = w_18 = None
        tmp_32 = torch.nn.functional.dropout(tmp_31, 0.0, False, False);  tmp_31 = None
        conv2d_3 = torch.conv2d(in_0, w_11, w_10, (4, 4), (3, 3), (1, 1), 1);  in_0 = w_11 = w_10 = None
        tmp_34 = torch.nn.functional.relu(conv2d_3, inplace = True);  conv2d_3 = None
        conv2d_4 = torch.conv2d(tmp_34, w_13, w_12, (2, 2), (1, 1), (1, 1), 1);  tmp_34 = w_13 = w_12 = None
        tmp_36 = torch.nn.functional.relu(conv2d_4, inplace = True);  conv2d_4 = None
        conv2d_5 = torch.conv2d(tmp_36, w_15, w_14, (2, 2), (1, 1), (1, 1), 1);  tmp_36 = w_15 = w_14 = None
        tmp_38 = conv2d_5.flatten(2);  conv2d_5 = None
        tmp_39 = tmp_38.transpose(1, 2);  tmp_38 = None
        tmp_40 = w_17.expand(1, -1, -1);  w_17 = None
        tmp_41 = torch.cat((tmp_40, tmp_39), dim = 1);  tmp_40 = tmp_39 = None
        tmp_42 = tmp_41 + w_19;  tmp_41 = w_19 = None
        tmp_43 = torch.nn.functional.dropout(tmp_42, 0.0, False, False);  tmp_42 = None
        tmp_44 = torch.nn.functional.layer_norm(tmp_32, (224,), w_3, w_2, 1e-06);  w_3 = w_2 = None
        linear = torch.nn.functional.linear(tmp_44, w_1, w_0);  tmp_44 = w_1 = w_0 = None
        return (linear, tmp_32, tmp_43)
        
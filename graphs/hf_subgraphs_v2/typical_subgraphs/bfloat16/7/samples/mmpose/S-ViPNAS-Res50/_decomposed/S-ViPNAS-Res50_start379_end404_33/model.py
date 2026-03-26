import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26):
        tmp_25 = in_26.view(256, 608, 48)
        tmp_26 = tmp_25.unsqueeze(1);  tmp_25 = None
        conv2d = torch.conv2d(in_26, in_7, in_6, (1, 1), (0, 0), (1, 1), 1);  in_7 = in_6 = None
        tmp_28 = conv2d.view(256, 1, 48);  conv2d = None
        tmp_29 = torch.nn.functional.softmax(tmp_28, 2, _stacklevel = 5);  tmp_28 = None
        tmp_30 = tmp_29.unsqueeze(-1);  tmp_29 = None
        matmul = torch.matmul(tmp_26, tmp_30);  tmp_26 = tmp_30 = None
        tmp_32 = matmul.view(256, 608, 1, 1);  matmul = None
        conv2d_1 = torch.conv2d(tmp_32, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  tmp_32 = in_1 = in_0 = None
        tmp_34 = torch.nn.functional.layer_norm(conv2d_1, (38, 1, 1), in_3, in_2, 1e-05);  conv2d_1 = in_3 = in_2 = None
        tmp_35 = torch.nn.functional.relu(tmp_34, inplace = True);  tmp_34 = None
        conv2d_2 = torch.conv2d(tmp_35, in_5, in_4, (1, 1), (0, 0), (1, 1), 1);  tmp_35 = in_5 = in_4 = None
        tmp_37 = in_26 + conv2d_2;  in_26 = conv2d_2 = None
        tmp_37 += in_25;  tmp_38 = tmp_37;  tmp_37 = in_25 = None
        tmp_39 = torch.nn.functional.relu(tmp_38, inplace = True);  tmp_38 = None
        tmp_40 = torch.conv_transpose2d(tmp_39, in_8, None, (2, 2), (1, 1), (0, 0), 16, (1, 1));  tmp_39 = in_8 = None
        tmp_41 = torch.nn.functional.batch_norm(tmp_40, in_9, in_10, in_12, in_11, False, 0.1, 1e-05);  tmp_40 = in_9 = in_10 = in_12 = in_11 = None
        tmp_42 = torch.nn.functional.relu(tmp_41, inplace = True);  tmp_41 = None
        tmp_43 = torch.conv_transpose2d(tmp_42, in_13, None, (2, 2), (1, 1), (0, 0), 16, (1, 1));  tmp_42 = in_13 = None
        tmp_44 = torch.nn.functional.batch_norm(tmp_43, in_14, in_15, in_17, in_16, False, 0.1, 1e-05);  tmp_43 = in_14 = in_15 = in_17 = in_16 = None
        tmp_45 = torch.nn.functional.relu(tmp_44, inplace = True);  tmp_44 = None
        tmp_46 = torch.conv_transpose2d(tmp_45, in_18, None, (2, 2), (1, 1), (0, 0), 16, (1, 1));  tmp_45 = in_18 = None
        tmp_47 = torch.nn.functional.batch_norm(tmp_46, in_19, in_20, in_22, in_21, False, 0.1, 1e-05);  tmp_46 = in_19 = in_20 = in_22 = in_21 = None
        tmp_48 = torch.nn.functional.relu(tmp_47, inplace = True);  tmp_47 = None
        conv2d_3 = torch.conv2d(tmp_48, in_24, in_23, (1, 1), (0, 0), (1, 1), 1);  tmp_48 = in_24 = in_23 = None
        return (conv2d_3,)
        
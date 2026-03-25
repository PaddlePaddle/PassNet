import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor, in_15 : torch.Tensor, in_16 : torch.Tensor, in_17 : torch.Tensor, in_18 : torch.Tensor, in_19 : torch.Tensor, in_20 : torch.Tensor, in_21 : torch.Tensor, in_22 : torch.Tensor, in_23 : torch.Tensor, in_24 : torch.Tensor, in_25 : torch.Tensor, in_26 : torch.Tensor, in_27 : torch.Tensor, in_28 : torch.Tensor):
        tmp_28 = torch.nn.functional.batch_norm(in_28, in_22, in_23, in_25, in_24, False, 0.1, 1e-05);  in_22 = in_23 = in_25 = in_24 = None
        conv2d = torch.conv2d(tmp_28, in_9, in_8, (1, 1), (0, 0), (1, 1), 1);  in_9 = in_8 = None
        tmp_30 = torch.nn.functional.gelu(conv2d);  conv2d = None
        conv2d_1 = torch.conv2d(tmp_30, in_3, in_2, (1, 1), (2, 2), (1, 1), 16);  in_3 = in_2 = None
        conv2d_2 = torch.conv2d(conv2d_1, in_1, in_0, (1, 1), (9, 9), (3, 3), 16);  conv2d_1 = in_1 = in_0 = None
        conv2d_3 = torch.conv2d(conv2d_2, in_5, in_4, (1, 1), (0, 0), (1, 1), 1);  conv2d_2 = in_5 = in_4 = None
        tmp_34 = tmp_30 * conv2d_3;  tmp_30 = conv2d_3 = None
        conv2d_4 = torch.conv2d(tmp_34, in_7, in_6, (1, 1), (0, 0), (1, 1), 1);  tmp_34 = in_7 = in_6 = None
        tmp_36 = conv2d_4 + tmp_28;  conv2d_4 = tmp_28 = None
        tmp_37 = in_10.unsqueeze(-1);  in_10 = None
        tmp_38 = tmp_37.unsqueeze(-1);  tmp_37 = None
        tmp_39 = tmp_38 * tmp_36;  tmp_38 = tmp_36 = None
        tmp_40 = in_28 + tmp_39;  in_28 = tmp_39 = None
        tmp_41 = torch.nn.functional.batch_norm(tmp_40, in_18, in_19, in_21, in_20, False, 0.1, 1e-05);  in_18 = in_19 = in_21 = in_20 = None
        conv2d_5 = torch.conv2d(tmp_41, in_14, in_13, (1, 1), (0, 0), (1, 1), 1);  tmp_41 = in_14 = in_13 = None
        conv2d_6 = torch.conv2d(conv2d_5, in_12, in_11, (1, 1), (1, 1), (1, 1), 128);  conv2d_5 = in_12 = in_11 = None
        tmp_44 = torch.nn.functional.gelu(conv2d_6);  conv2d_6 = None
        tmp_45 = torch.nn.functional.dropout(tmp_44, 0.0, False, False);  tmp_44 = None
        conv2d_7 = torch.conv2d(tmp_45, in_16, in_15, (1, 1), (0, 0), (1, 1), 1);  tmp_45 = in_16 = in_15 = None
        tmp_47 = torch.nn.functional.dropout(conv2d_7, 0.0, False, False);  conv2d_7 = None
        tmp_48 = in_17.unsqueeze(-1);  in_17 = None
        tmp_49 = tmp_48.unsqueeze(-1);  tmp_48 = None
        tmp_50 = tmp_49 * tmp_47;  tmp_49 = tmp_47 = None
        tmp_51 = tmp_40 + tmp_50;  tmp_40 = tmp_50 = None
        tmp_52 = tmp_51.flatten(2);  tmp_51 = None
        tmp_53 = tmp_52.transpose(1, 2);  tmp_52 = None
        tmp_54 = torch.nn.functional.layer_norm(tmp_53, (16,), in_27, in_26, 1e-06);  tmp_53 = in_27 = in_26 = None
        tmp_55 = tmp_54.view(64, 56, 56, 16);  tmp_54 = None
        tmp_56 = tmp_55.permute(0, 3, 1, 2);  tmp_55 = None
        return (tmp_56,)
        
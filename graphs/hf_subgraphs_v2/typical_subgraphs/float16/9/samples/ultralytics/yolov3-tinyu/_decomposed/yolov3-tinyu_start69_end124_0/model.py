import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_4 = torch.nn.functional.silu(in_0, inplace = True);  in_0 = None
        conv2d = torch.conv2d(tmp_4, w_1, w_0, (1, 1), (0, 0), (1, 1), 1);  tmp_4 = w_1 = w_0 = None
        tmp_6 = torch.cat((in_1, conv2d), 1);  in_1 = conv2d = None
        tmp_7 = in_2.view(1, 144, -1);  in_2 = None
        tmp_8 = tmp_6.view(1, 144, -1)
        tmp_9 = torch.cat([tmp_7, tmp_8], 2);  tmp_7 = tmp_8 = None
        tmp_10 = w_3[0]
        tmp_11 = w_3[1];  w_3 = None
        tmp_12 = torch.arange(end = 40, device = device(type='cuda', index=0), dtype = torch.float32)
        tmp_13 = tmp_12 + 0.5;  tmp_12 = None
        tmp_14 = torch.arange(end = 40, device = device(type='cuda', index=0), dtype = torch.float32)
        tmp_15 = tmp_14 + 0.5;  tmp_14 = None
        meshgrid = torch.functional.meshgrid(tmp_15, tmp_13, indexing = 'ij');  tmp_15 = tmp_13 = None
        tmp_17 = meshgrid[0]
        tmp_18 = meshgrid[1];  meshgrid = None
        tmp_19 = torch.stack((tmp_18, tmp_17), -1);  tmp_18 = tmp_17 = None
        tmp_20 = tmp_19.view(-1, 2);  tmp_19 = None
        _local_scalar_dense = torch.ops.aten._local_scalar_dense(tmp_10);  tmp_10 = None
        tmp_22 = torch.full((1600, 1), _local_scalar_dense, dtype = torch.float32, device = device(type='cuda', index=0));  _local_scalar_dense = None
        tmp_23 = torch.arange(end = 20, device = device(type='cuda', index=0), dtype = torch.float32)
        tmp_24 = tmp_23 + 0.5;  tmp_23 = None
        tmp_25 = torch.arange(end = 20, device = device(type='cuda', index=0), dtype = torch.float32)
        tmp_26 = tmp_25 + 0.5;  tmp_25 = None
        meshgrid_1 = torch.functional.meshgrid(tmp_26, tmp_24, indexing = 'ij');  tmp_26 = tmp_24 = None
        tmp_28 = meshgrid_1[0]
        tmp_29 = meshgrid_1[1];  meshgrid_1 = None
        tmp_30 = torch.stack((tmp_29, tmp_28), -1);  tmp_29 = tmp_28 = None
        tmp_31 = tmp_30.view(-1, 2);  tmp_30 = None
        _local_scalar_dense_1 = torch.ops.aten._local_scalar_dense(tmp_11);  tmp_11 = None
        tmp_33 = torch.full((400, 1), _local_scalar_dense_1, dtype = torch.float32, device = device(type='cuda', index=0));  _local_scalar_dense_1 = None
        tmp_34 = torch.cat([tmp_20, tmp_31]);  tmp_20 = tmp_31 = None
        tmp_35 = torch.cat([tmp_22, tmp_33]);  tmp_22 = tmp_33 = None
        tmp_36 = tmp_34.transpose(0, 1);  tmp_34 = None
        tmp_37 = tmp_35.transpose(0, 1);  tmp_35 = None
        split = tmp_9.split((64, 80), 1);  tmp_9 = None
        tmp_39 = split[0]
        tmp_40 = split[1];  split = None
        tmp_41 = tmp_39.view(1, 4, 16, 2000);  tmp_39 = None
        tmp_42 = tmp_41.transpose(2, 1);  tmp_41 = None
        tmp_43 = tmp_42.softmax(1);  tmp_42 = None
        conv2d_1 = torch.conv2d(tmp_43, w_2, None, (1, 1), (0, 0), (1, 1), 1);  tmp_43 = w_2 = None
        tmp_45 = conv2d_1.view(1, 4, 2000);  conv2d_1 = None
        tmp_46 = tmp_36.unsqueeze(0)
        chunk = tmp_45.chunk(2, 1);  tmp_45 = None
        tmp_48 = chunk[0]
        tmp_49 = chunk[1];  chunk = None
        tmp_50 = tmp_46 - tmp_48;  tmp_48 = None
        tmp_51 = tmp_46 + tmp_49;  tmp_46 = tmp_49 = None
        tmp_52 = tmp_50 + tmp_51
        tmp_53 = tmp_52 / 2;  tmp_52 = None
        tmp_54 = tmp_51 - tmp_50;  tmp_51 = tmp_50 = None
        tmp_55 = torch.cat((tmp_53, tmp_54), 1);  tmp_53 = tmp_54 = None
        tmp_56 = tmp_55 * tmp_37;  tmp_55 = None
        tmp_57 = tmp_40.sigmoid();  tmp_40 = None
        tmp_58 = torch.cat((tmp_56, tmp_57), 1);  tmp_56 = tmp_57 = None
        return (tmp_6, tmp_36, tmp_37, tmp_58)
        
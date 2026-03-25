class Program_weight_tensor_meta_query_layer:
	name = "in_0"
	original_name = "query_layer"
	shape = [1, 1, 256, 16]
	dtype = "torch.float16"
	device = "cuda:0"
	mean = -0.005
	std = 0.014


class Program_weight_tensor_meta_transpose_4:
	name = "in_1"
	original_name = "transpose_4"
	shape = [1, 1, 16, 4]
	dtype = "torch.float16"
	device = "cuda:0"
	mean = 0.003
	std = 0.017


class Program_weight_tensor_meta_value_layer:
	name = "in_2"
	original_name = "value_layer"
	shape = [1, 1, 4, 16]
	dtype = "torch.float16"
	device = "cuda:0"
	mean = -0.001
	std = 0.015

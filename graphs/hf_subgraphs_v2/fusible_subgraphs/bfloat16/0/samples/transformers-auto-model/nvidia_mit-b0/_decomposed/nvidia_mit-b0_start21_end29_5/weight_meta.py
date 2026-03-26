class Program_weight_tensor_meta_query_layer:
	name = "in_0"
	original_name = "query_layer"
	shape = [1, 1, 16384, 32]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = -0.004
	std = 0.019


class Program_weight_tensor_meta_transpose_4:
	name = "in_1"
	original_name = "transpose_4"
	shape = [1, 1, 32, 256]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = -0.001
	std = 0.022


class Program_weight_tensor_meta_value_layer:
	name = "in_2"
	original_name = "value_layer"
	shape = [1, 1, 256, 32]
	dtype = "torch.bfloat16"
	device = "cuda:0"
	mean = 0.010
	std = 0.020

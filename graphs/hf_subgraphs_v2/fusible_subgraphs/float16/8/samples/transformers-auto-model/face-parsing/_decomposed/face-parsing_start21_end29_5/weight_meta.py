class Program_weight_tensor_meta_query_layer:
	name = "in_0"
	original_name = "query_layer"
	shape = [64, 1, 16384, 64]
	dtype = "torch.float16"
	device = "cuda:0"
	mean = -0.077
	std = 0.340


class Program_weight_tensor_meta_transpose_4:
	name = "in_1"
	original_name = "transpose_4"
	shape = [64, 1, 64, 256]
	dtype = "torch.float16"
	device = "cuda:0"
	mean = 0.083
	std = 0.154


class Program_weight_tensor_meta_value_layer:
	name = "in_2"
	original_name = "value_layer"
	shape = [64, 1, 256, 64]
	dtype = "torch.float16"
	device = "cuda:0"
	mean = -0.001
	std = 0.027

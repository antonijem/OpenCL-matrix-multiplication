__kernel void matrix_mult(__global float* A, __global float* B, __global float* C, int m, int n, int k){
	int i = get_global_id(0);
	int j = get_global_id(1);
	float sum = 0.0f;
	for(int l = 0; l < k; l++){
		sum += A[i * k + l] * B[l * n + j];
	}
	C[i * n + j] = sum; 
}
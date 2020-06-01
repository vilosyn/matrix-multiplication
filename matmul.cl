__kernel void addMul(double a, double b, __local double *c) { *c += a * b; }

/*
double addMul(const double a, const double b) {
      return a * b;
}
*/

__kernel void matrixVectorMul(__global double *resultVector,
                              __global double *matrixA,
                              __global double *matrixB, int width_A,
                              int height_A) {
  // int i = get_global_id(0);
  // int j = get_global_id(1);
  __local double value;
  int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
  int j = get_local_size(1) * get_group_id(1) + get_local_id(1);
  if (i < width_A && j < width_A) {
    value = 0;
    for (unsigned int k = 0; k < height_A; ++k)
	    addMul(matrixA[j * height_A + k], matrixB[k * width_A + i], &value);
    // value += addMul(matrixA[j * height_A + k], matrixB[k * width_A + i]);
    // value += matrixA[j * height_A + k] * matrixB[k * width_A + i];

    resultVector[j * width_A + i] = value;
  }
}

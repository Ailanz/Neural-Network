
// NeuralNetwork.teach.GpuTrainer
extern "C" __global__  void CalculateChangeDeltaAndError( double* error, int errorLen0,  double* inputs, int inputsLen0, int inputsLen1,  double* previousChangeDelta, int previousChangeDeltaLen0, int previousChangeDeltaLen1,  double* weights, int weightsLen0, int weightsLen1,  double* changeDelta, int changeDeltaLen0, int changeDeltaLen1,  double* backPropError, int backPropErrorLen0, int backPropErrorLen1);

// NeuralNetwork.teach.GpuTrainer
extern "C" __global__  void CalculateChangeDeltaAndError( double* error, int errorLen0,  double* inputs, int inputsLen0, int inputsLen1,  double* previousChangeDelta, int previousChangeDeltaLen0, int previousChangeDeltaLen1,  double* weights, int weightsLen0, int weightsLen1,  double* changeDelta, int changeDeltaLen0, int changeDeltaLen1,  double* backPropError, int backPropErrorLen0, int backPropErrorLen1)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	if (y < 1024)
	{
		changeDelta[(y) * changeDeltaLen1 + ( x)] = error[(y)] * inputs[(y) * inputsLen1 + ( x)] * 0.3 + previousChangeDelta[(y) * previousChangeDeltaLen1 + ( x)] * 0.05;
		backPropError[(y) * backPropErrorLen1 + ( x)] = error[(y)] * weights[(y) * weightsLen1 + ( x)];
	}
}


// NeuralNetwork.teach.GpuTrainer
extern "C" __global__  void CalculateChangeDeltaAndError( double* error, int errorLen0,  double* inputs, int inputsLen0, int inputsLen1,  double* previousChangeDelta, int previousChangeDeltaLen0, int previousChangeDeltaLen1,  double* weights, int weightsLen0, int weightsLen1,  double* changeDelta, int changeDeltaLen0, int changeDeltaLen1,  double* backPropError, int backPropErrorLen0, int backPropErrorLen1);

// NeuralNetwork.teach.GpuTrainer
extern "C" __global__  void CalculateChangeDeltaAndError( double* error, int errorLen0,  double* inputs, int inputsLen0, int inputsLen1,  double* previousChangeDelta, int previousChangeDeltaLen0, int previousChangeDeltaLen1,  double* weights, int weightsLen0, int weightsLen1,  double* changeDelta, int changeDeltaLen0, int changeDeltaLen1,  double* backPropError, int backPropErrorLen0, int backPropErrorLen1)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	if (x < 1024)
	{
		changeDelta[(x) * changeDeltaLen1 + ( y)] = error[(x)] * inputs[(x) * inputsLen1 + ( y)] * 0.3 + previousChangeDelta[(x) * previousChangeDeltaLen1 + ( y)] * 0.05;
		backPropError[(x) * backPropErrorLen1 + ( y)] = error[(x)] * weights[(x) * weightsLen1 + ( y)];
	}
}

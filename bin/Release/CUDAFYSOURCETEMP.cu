
// NeuralNetwork.teach.CudaTrainer
extern "C" __global__  void CalculateChangeDelta(double error, double input, double learnRate, double previousChangeDelta, double momentum,  double* c, int cLen0);
// NeuralNetwork.teach.CudaTrainer
extern "C" __global__  void Multiply(double a, double b,  double* c, int cLen0);

// NeuralNetwork.teach.CudaTrainer
extern "C" __global__  void CalculateChangeDelta(double error, double input, double learnRate, double previousChangeDelta, double momentum,  double* c, int cLen0)
{
	c[(0)] = error * input * learnRate + previousChangeDelta * momentum;
}
// NeuralNetwork.teach.CudaTrainer
extern "C" __global__  void Multiply(double a, double b,  double* c, int cLen0)
{
	c[(0)] = a * b;
}

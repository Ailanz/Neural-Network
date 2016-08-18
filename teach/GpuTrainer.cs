using NeuralNetwork.neuron;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace NeuralNetwork.teach
{
    public class GpuTrainer
    {
        Network network;
        static Random random = new Random();
        const int RANDOM_SAMPLE = 35;
        public static int counter = 0;
        int maxRepetition = -1;
        int minRepetition = 0;
        const double learnRate = 0.30;
        const double momentum = 0.05;
        public const int N = 1024;

        //Cudafy
        GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
        CudafyModule km = CudafyTranslator.Cudafy();

        public delegate void Callback(Network network, List<double[]> inputs, List<double[]> targets, double errorRate, int repetition);


        public GpuTrainer(Network network)
        {
            this.network = network;
            gpu.LoadModule(km);
        }

        public double Train(List<double[]> inputs, List<double[]> targets, double precision, Callback callback)
        {
            double sumError = Double.MaxValue;
            int repetition = 0;
            while (sumError > precision || (maxRepetition-- > 0 || maxRepetition == -1) || repetition < minRepetition)
            {
                sumError = 0;
                for (int i = 0; i < inputs.Count(); i++)
                {
                    sumError += TrainNetworkAndReturnErrorRate(inputs[i], targets[i]);
                }

                sumError = sumError / inputs.Count();

                if (callback != null)
                {
                    callback(this.network, inputs, targets, sumError, repetition);
                }

                repetition++;
            }
            Console.WriteLine("Stopped at: " + repetition + " with error: " + sumError);
            return sumError;
        }

        public double TrainNetworkAndReturnErrorRate(double[] inputs, double[] targets)
        {
            this.network.SetInputs(inputs);
            Neuron[] outputs = this.network.GetOutputNeurons();

            if (outputs.Length != targets.Length)
            {
                throw new Exception(String.Format("Output length mismatch {0} - {1}", outputs.Length, targets.Length));
            }

            //Input and Output layer neurons are always non-null
            TrainOutputLayer(targets);

            for (int i = this.network.hiddenLayerNeuronsList.Count - 1; i >= 0; i--)
            {
                TrainLayerNeurons(this.network.hiddenLayerNeuronsList[i]);
            }

            TrainLayerNeurons(this.network.GetInputNeurons());

            return EstimateErrorRate(inputs, targets);
        }


        public void TrainOutputLayer(double[] targets)
        {
            Neuron[] neuronsToTrain = this.network.GetOutputNeurons();
            int numNeurons = neuronsToTrain.Length;             

                double[] errors = new Double[neuronsToTrain.Length];
                double[,] inputs = new Double[numNeurons, neuronsToTrain[0].GetInputs().Length];
                double[,] prevChangeDeltas = new Double[numNeurons, neuronsToTrain[0].previousChangeDelta.Length];
                double[,] weights = new Double[numNeurons, neuronsToTrain[0].weights.Length];
                double[,] changeDeltaResult = new Double[numNeurons, neuronsToTrain[0].weights.Length];
                double[,] backPropErrorResult = new Double[numNeurons, neuronsToTrain[0].weights.Length];

                for (int y = 0; y < neuronsToTrain.Length; y++)
                {
                    Neuron neuron = neuronsToTrain[y];
                    double output = neuron.GetOutput();
                    errors[y] = neuronsToTrain[0].activationFunction.GetSquashFunction(output) * (targets[y] - neuron.GetOutput());
                    neuron.backPropogationError = 0;
                    for (int x = 0; x < neuron.GetInputs().Length; x++)
                    {
                        inputs[y, x] = neuron.GetInputs()[x];
                    }

                    for (int x = 0; x < neuron.previousChangeDelta.Length; x++)
                    {
                        prevChangeDeltas[y, x] = neuron.previousChangeDelta[x];
                    }

                    for (int x = 0; x < neuron.weights.Length; x++)
                    {
                        weights[y, x] = neuron.weights[x];
                    }
                }

//Use host arrays instead of size
                double[] dev_errors = gpu.Allocate<double>(errors);
                double[,] dev_inputs = gpu.Allocate<double>(numNeurons, inputs.Length);
                double[,] dev_prevChangeDelta = gpu.Allocate<double>(numNeurons, neuronsToTrain[0].previousChangeDelta.Length);
                double[,] dev_weights = gpu.Allocate<double>(numNeurons, neuronsToTrain[0].weights.Length);
                double[,] dev_changeDeltaResult = gpu.Allocate<double>(changeDeltaResult);
                double[,] dev_backPropErrorResult = gpu.Allocate<double>(backPropErrorResult);

//Copy the errors!
                gpu.CopyToDevice(inputs, dev_inputs);
                gpu.CopyToDevice(prevChangeDeltas, dev_prevChangeDelta);
                gpu.CopyToDevice(weights, dev_weights);

                // figure out how to launch Y x X Jobs
                gpu.Launch().CalculateChangeDeltaAndError(dev_errors, dev_inputs, dev_prevChangeDelta, dev_weights, dev_changeDeltaResult, dev_backPropErrorResult);

                gpu.CopyFromDevice(dev_changeDeltaResult, changeDeltaResult);
                gpu.CopyFromDevice(dev_backPropErrorResult, backPropErrorResult);
                for (int y = 0; y < neuronsToTrain.Length; y++)
                {
                    Neuron neuron = neuronsToTrain[y];
                    for (int x = 0; x < neuron.weights.Length; x++)
                    {
                        //index is incorrect? should use X
                        neuron.neuronInputs[y].backPropogationError += backPropErrorResult[y,x];
                        neuron.weights[y] += changeDeltaResult[y,x];
                        neuron.previousChangeDelta[y] = changeDeltaResult[y,x];
                        //Propogate the error back to previous layer neuron
                    }
                    ModifyBias(neuron, errors[y]);
                    neuron.hasUpdated = true;
                }
                //Modify Bias
               
                // free dev_error
                gpu.Free(dev_inputs);
                gpu.Free(dev_prevChangeDelta);
                gpu.Free(dev_weights);
                gpu.Free(dev_changeDeltaResult);
                gpu.Free(dev_backPropErrorResult);
            
        }

        [Cudafy]
        public static void CalculateChangeDeltaAndError(GThread thread, double[] error, double[,] inputs, double[,] previousChangeDelta, double[,] weights, double[,] changeDelta, double[,] backPropError)
        {
            int tidx = thread.blockIdx.x;
            int tidy = thread.blockIdx.y;
            if (tidx < N)
            {
                //swap x and y
                changeDelta[tidx, tidy] = (error[tidx] * inputs[tidx, tidy]) * learnRate + previousChangeDelta[tidx, tidy] * momentum;
                backPropError[tidx, tidy] = error[tidx] * weights[tidx, tidy];
            }
        }

        public void TrainLayerNeurons(Neuron[] neurons)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                Neuron neuron = neurons[i];
                double output = neuron.GetOutput(); //0.14
                double error = neuron.activationFunction.GetSquashFunction(output) * neurons[i].backPropogationError;
                neuron.backPropogationError = 0;
                double[] inputs = neuron.GetInputs(); //0.15

                for (int j = 0; j < neuron.weights.Length; j++)
                {
                    double changeDelta = (error * inputs[j]) * learnRate + neuron.previousChangeDelta[j] * momentum;
                    //Modify each weight
                    if (!neuron.isInputLayer())
                    {
                        double backPropError = error * neuron.weights[j];
                        neuron.neuronInputs[j].backPropogationError += backPropError;
                    }
                    neuron.weights[j] += changeDelta;
                    neuron.previousChangeDelta[j] = changeDelta;
                    //Propogate the error back to previous layer neuron
                }
                //Modify Bias
                ModifyBias(neuron, error);
                neuron.hasUpdated = true;
            }
        }

        public double EstimateErrorRate(double[] inputs, double[] targets)
        {
            double totalError = 0;
            this.network.SetInputs(inputs);
            double[] curOutputs = this.network.GetOutputsAsDoubleArray();
            for (int i = 0; i < curOutputs.Length; i++)
            {
                double result = Math.Sqrt(Math.Abs(targets[i] * targets[i] - curOutputs[i] * curOutputs[i]));
                totalError += result;
            }
            return Math.Round(totalError / curOutputs.Length, 4);
        }

        public void ModifyBias(Neuron neuron, double error)
        {
            neuron.BIAS_WEIGHT += error * learnRate;
        }

        public void SetMaxReptition(int i)
        {
            this.maxRepetition = i;
        }
        public void SetMinReptition(int i)
        {
            this.minRepetition = i;
        }

    }
}

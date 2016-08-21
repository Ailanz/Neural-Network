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
using System.Diagnostics;

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
        const double learnRate = 0.35;
        const double momentum = 0.05;


        //OutputLayer Array
        double[] errors = null;
        double[,] inputs = null;
        double[,] prevChangeDeltas = null;
        double[,] weights = null;
        double[,] changeDeltaResult = null;
        double[,] backPropErrorResult = null;

        double[] dev_errors = null;
        double[,] dev_inputs = null;
        double[,] dev_prevChangeDelta = null;
        double[,] dev_weights = null;
        double[,] dev_changeDeltaResult = null;
        double[,] dev_backPropErrorResult = null;

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
                Stopwatch stopwatch = Stopwatch.StartNew();
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
                Console.WriteLine("Iteration: " + stopwatch.ElapsedMilliseconds);

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
            Stopwatch stopwatch = Stopwatch.StartNew();
            Neuron[] neuronsToTrain = this.network.GetOutputNeurons();
            int numNeurons = neuronsToTrain.Length;
            int lengthOfInputsAndWeights = neuronsToTrain[0].weights.Length;

            if (errors == null)
            {
                errors = new double[neuronsToTrain.Length];
                inputs = new double[numNeurons, lengthOfInputsAndWeights];
                prevChangeDeltas = new double[numNeurons, lengthOfInputsAndWeights];
                weights = new double[numNeurons, lengthOfInputsAndWeights];
                changeDeltaResult = new double[numNeurons, lengthOfInputsAndWeights];
                backPropErrorResult = new double[numNeurons, lengthOfInputsAndWeights];
            }

            //Console.WriteLine("1: " + stopwatch.ElapsedMilliseconds);

            Parallel.For(0, neuronsToTrain.Length, y =>
            {
                Neuron neuron = neuronsToTrain[y];
                double output = neuron.GetOutput();

                errors[y] = neuronsToTrain[0].activationFunction.GetSquashFunction(output) * (targets[y] - neuron.GetOutput());
                neuron.backPropogationError = 0;

                double[] neuronInputs = neuron.GetInputs();

                Parallel.Invoke(() =>
                {
                    for (int x = 0; x < neuronInputs.Length; x++)
                    {
                        //Run in Parallel
                        prevChangeDeltas[y, x] = neuron.previousChangeDelta[x];
                    }
                },
                () =>
                {
                    for (int x = 0; x < neuronInputs.Length; x++)
                    {
                        //Run in Parallel
                        inputs[y, x] = neuronInputs[x];
                    }
                },
                () =>
                {
                    for (int x = 0; x < neuronInputs.Length; x++)
                    {
                        //Run in Parallel                    
                        weights[y, x] = neuron.weights[x];
                    }
                }
                );
            });

            //Console.WriteLine("2: " + stopwatch.ElapsedMilliseconds);
            //stopwatch.Restart();

            //Use host arrays instead of size
            if (dev_errors == null)
            {
                dev_errors = gpu.Allocate<double>(errors);
                dev_inputs = gpu.Allocate<double>(numNeurons, inputs.Length);
                dev_prevChangeDelta = gpu.Allocate<double>(numNeurons, neuronsToTrain[0].previousChangeDelta.Length);
                dev_weights = gpu.Allocate<double>(numNeurons, neuronsToTrain[0].weights.Length);
                dev_changeDeltaResult = gpu.Allocate<double>(changeDeltaResult);
                dev_backPropErrorResult = gpu.Allocate<double>(backPropErrorResult);
            }
            // Console.WriteLine("2.5: " + stopwatch.ElapsedMilliseconds);
            //stopwatch.Restart();

            //Run in Parallel
            gpu.CopyToDevice(errors, dev_errors);
            gpu.CopyToDevice(inputs, dev_inputs);
            gpu.CopyToDevice(prevChangeDeltas, dev_prevChangeDelta);
            gpu.CopyToDevice(weights, dev_weights);
            //Console.WriteLine("3: " + stopwatch.ElapsedMilliseconds);
            //stopwatch.Restart();

            // figure out how to launch Y x X Jobs
            gpu.Launch().CalculateChangeDeltaAndError(dev_errors, dev_inputs, dev_prevChangeDelta, dev_weights, dev_changeDeltaResult, dev_backPropErrorResult);

            gpu.CopyFromDevice(dev_changeDeltaResult, changeDeltaResult);
            //Console.WriteLine("4: " + stopwatch.ElapsedMilliseconds);

            gpu.CopyFromDevice(dev_backPropErrorResult, backPropErrorResult);
            //Console.WriteLine("4.1: " + stopwatch.ElapsedMilliseconds);
           // stopwatch.Restart();

            Parallel.For(0, neuronsToTrain.Length, y =>
            {
                Neuron neuron = neuronsToTrain[y];

                Parallel.Invoke(() =>
                {
                    for (int x = 0; x < neuron.weights.Length; x++)
                    {
                        //Run in Parallel
                        neuron.neuronInputs[x].backPropogationError += backPropErrorResult[y, x];
                    }
                },
                () =>
                {
                    for (int x = 0; x < neuron.weights.Length; x++)
                    {
                        neuron.weights[x] += changeDeltaResult[y, x];
                    }
                },
                () =>
                {
                    for (int x = 0; x < neuron.weights.Length; x++)
                    {
                        //Run in Parallel                      
                        neuron.previousChangeDelta[x] = changeDeltaResult[y, x];
                        //Propogate the error back to previous layer neuron
                    }
                },
                () =>
                {
                    ModifyBias(neuron, errors[y]);
                    neuron.hasUpdated = true;
                }
                );
            });
            //Modify Bias
            //Console.WriteLine("5: " + stopwatch.ElapsedMilliseconds);

            // free dev_error
            //gpu.Free(dev_errors);
            //gpu.Free(dev_inputs);
            //gpu.Free(dev_prevChangeDelta);
            //gpu.Free(dev_weights);
            //gpu.Free(dev_changeDeltaResult);
            //gpu.Free(dev_backPropErrorResult);

            //Console.WriteLine("6: " + stopwatch.ElapsedMilliseconds);
            stopwatch.Stop();
        }

        [Cudafy]
        public static void CalculateChangeDeltaAndError(GThread thread, double[] error, double[,] inputs, double[,] previousChangeDelta, double[,] weights, double[,] changeDelta, double[,] backPropError)
        {
            int tidy = thread.blockIdx.x;
            int tidx = thread.blockIdx.y;

            changeDelta[tidx, tidy] = (error[tidx] * inputs[tidx, tidy]) * learnRate + previousChangeDelta[tidx, tidy] * momentum;
            backPropError[tidx, tidy] = error[tidx] * weights[tidx, tidy];

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

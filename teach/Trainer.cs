using NeuralNetwork.activationFunction;
using NeuralNetwork.neuron;
using NeuralNetwork.normalizer;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.teach
{
    //Trainer uses back propogation method to train the network
    public class Trainer
    {
        Network network;
        double learnRate = 0.30;
        double momentum = 0.1;
        static Random random = new Random();
        const int RANDOM_SAMPLE = 35;

        public delegate void Callback(Network network, List<double[]> inputs, List<double[]> targets, double errorRate, int repetition);

        public Trainer(Network network)
        {
            this.network = network;
        }
        public double Train(List<double[]> inputs, List<double[]> targets, double precision)
        {
            return Train(inputs, targets, precision, null);
        }

        public double Train(List<double[]> inputs, List<double[]> targets, double precision, Callback callback)
        {
            double sumError = Double.MaxValue;
            int repetition = 0;
            while (sumError > precision)
            {
                sumError = 0;
                for (int i = 0; i < inputs.Count(); i++)
                {
                    sumError += TrainNetworkAndReturnErrorRate(inputs[i], targets[i]);
                }

                sumError = sumError / inputs.Count();

                if (callback != null)
                {
                    callback(this.network, inputs, targets, sumError,repetition);
                }

                repetition++;
            }
            Console.WriteLine("Stopped at: " + repetition + " with error: " + sumError + ", Learn Rate: " + learnRate);
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
            for (int i = this.network.hiddenLayerNeuronsList.Count - 1; i > 0; i--)
            {
                TrainLayerNeurons(this.network.hiddenLayerNeuronsList[i]);
            }
            TrainLayerNeurons(this.network.GetInputNeurons());

            return EstimateErrorRate(inputs, targets);           
        }


        public void TrainOutputLayer(double[] targets)
        {
            int stepCount = 0;
            Neuron[] neurons = this.network.GetOutputNeurons();
            for (int i = 0; i < neurons.Length; i++)
            {
                double output = neurons[i].GetOutput(); //0.14
                double error = neurons[i].activationFunction.GetSquashFunction(output) * (targets[i] - output);
                double[] inputs = neurons[i].GetInputs(); //0.15

                for (int j = 0; j < neurons[i].weights.Length; j++)
                {
                    double changeDelta = (error * inputs[j]) * learnRate + neurons[i].previousChangeDelta[j] * momentum;
                    //Modify each weight
                    neurons[i].neuronInputs[j].backPropogationError += error * neurons[i].weights[j];

                    neurons[i].weights[j] += changeDelta;
                    stepCount++;
                    neurons[i].previousChangeDelta[j] = changeDelta;
                    //Propogate the error back to previous layer neuron
                }
                //Modify Bias
                ModifyBias(neurons[i], error);
            }
           // Console.WriteLine(neuronsToTrain.Length + " * " + neuronsToTrain[0].weights.Length + " = " + stepCount);

        }

        public void TrainLayerNeurons(Neuron[] neurons)
        {
            int stepCount = 0;
            for (int i = 0; i < neurons.Length; i++)
            {
                double output = neurons[i].GetOutput();
                double error = neurons[i].activationFunction.GetSquashFunction(output) * neurons[i].backPropogationError;
                neurons[i].backPropogationError = 0; //reset current neuron's error
                double[] inputs = neurons[i].GetInputs();

                for (int j = 0; j < neurons[i].weights.Length; j++)
                {
                    double changeDelta = (error * inputs[j]) * learnRate + neurons[i].previousChangeDelta[j] * momentum;
                    //Propogate the error back to previous layer neuron
                    if (!neurons[i].isInputLayer())
                    {
                        neurons[i].neuronInputs[j].backPropogationError += error * neurons[i].weights[j];
                    }
                    //Modify each weight
                    neurons[i].weights[j] += changeDelta;
                    neurons[i].previousChangeDelta[j] = changeDelta;
                    stepCount++;

                }
                //Modify Bias
                ModifyBias(neurons[i], error);
            }
            //Console.WriteLine(neurons.Length + " * " + neurons[0].weights.Length + " = " + stepCount);
        }

        public void ModifyBias(Neuron neuron, double error)
        {
            neuron.BIAS_WEIGHT += error * learnRate;
        }

        public double EstimateErrorRate(double[] inputs, double[] targets)
        {
            double totalError = 0;
            double[] curOutputs = this.network.GetOutputsAsDoubleArray();
            for (int i = 0; i < RANDOM_SAMPLE; i++)
            {
                int randomPick = random.Next(0, curOutputs.Length);
                double result = Math.Sqrt(Math.Abs(targets[randomPick] * targets[randomPick] - curOutputs[randomPick] * curOutputs[randomPick]));
                totalError += result;
            }
            return Math.Round(totalError / RANDOM_SAMPLE, 4);
        }

    }
}

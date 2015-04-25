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
    public class Trainer
    {
        Network network;
        static double LEARN_RATE = 0.30;
        static double MOMENTUM = 0.1;
        static Random random = new Random();
        const int RANDOM_SAMPLE = 5;
        public delegate void Callback(Network network, List<double[]> inputs, List<double[]> targets, int repetition);

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
            //precision in decimal
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            double errorRate = Double.MaxValue;
            int repetition = 0;
            while (errorRate > precision)
            {
                errorRate = 0;

                for (int i = 0; i < inputs.Count(); i++)
                {
                    errorRate += Train(inputs[i], targets[i]);
                }
                errorRate = errorRate / inputs.Count();
                if (repetition % 5 == 0)
                {
                    TimeSpan ts = stopWatch.Elapsed;
                    string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                    ts.Hours, ts.Minutes, ts.Seconds,
                    ts.Milliseconds / 10);
                    stopWatch.Restart();
                    
                    if(callback!=null)
                    {
                        callback(this.network, inputs, targets, repetition);
                    }

                    Console.WriteLine("Error Rate: " + errorRate + " Time: " + elapsedTime);
                    //Console.WriteLine("Error Rate: " + errorRate);
                }
                repetition++;
            }
            Console.WriteLine("Stopped at: " + repetition + " with error: " + errorRate);
            return errorRate;
        }

        public double Train(double[] inputs, double[] targets)
        {
            this.network.SetInputs(inputs);
            Neuron[] outputs = this.network.GetOutputNeurons();

            if (outputs.Length != targets.Length)
            {
                throw new Exception(String.Format("Output length mismatch {0} - {1}", outputs.Length, targets.Length));
            }

            //Train(targets, outputs);
            TrainOutputLayer(targets);
            if (this.network.GetSecondHiddenLayerNeurons() != null || this.network.GetSecondHiddenLayerNeurons().Length != 0)
            {
                TrainLayerNeurons(this.network.GetSecondHiddenLayerNeurons());
            }
            if (this.network.GetHiddenLayerNeurons() != null || this.network.GetHiddenLayerNeurons().Length != 0)
            {
                TrainLayerNeurons(this.network.GetHiddenLayerNeurons());
            }
            TrainLayerNeurons(this.network.GetInputNeurons());


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

        public void TrainOutputLayer(double[] targets)
        {
            int stepCount = 0;
            Neuron[] neuronsToTrain = this.network.GetOutputNeurons();
            for (int i = 0; i < neuronsToTrain.Length; i++)
            {
                Stopwatch stopWatch = new Stopwatch();
                stopWatch.Start();

                double output = neuronsToTrain[i].GetOutput(); //0.14
                double error = neuronsToTrain[i].activationFunction.GetSquashFunction(output) * (targets[i] - output);
                double[] inputs = neuronsToTrain[i].GetInputs(); //0.15


                for (int j = 0; j < neuronsToTrain[i].weights.Length; j++)
                {
                    double changeDelta = (error * inputs[j]) * LEARN_RATE +neuronsToTrain[i].previousChangeDelta[j] * MOMENTUM;
                    //Modify each weight
                    neuronsToTrain[i].neuronInputs[j].backPropogationError += error * neuronsToTrain[i].weights[j];

                    neuronsToTrain[i].weights[j] += changeDelta;
                    stepCount++;
                    neuronsToTrain[i].previousChangeDelta[j] = changeDelta;
                    //Propogate the error back to previous layer neuron
                }
                //Modify Bias
                ModifyBias(neuronsToTrain[i], error);
                TimeSpan ts = stopWatch.Elapsed;
                string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                ts.Hours, ts.Minutes, ts.Seconds,
                ts.Milliseconds / 10);
                //Console.WriteLine("FOR1 Time: " + elapsedTime + " : " + i);
                stopWatch.Restart();
            }
            Console.WriteLine(neuronsToTrain.Length + " * " + neuronsToTrain[0].weights.Length + " = " + stepCount);

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
                    double changeDelta = (error * inputs[j]) * LEARN_RATE + neurons[i].previousChangeDelta[j] * MOMENTUM;
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
        [Obsolete]
        public void TrainHiddenLayer(double error, Neuron[] neuronToTrain)
        {
            for (int i = 0; i < neuronToTrain.Length; i++)
            {
                double output = neuronToTrain[i].GetOutput();
                double hiddenError = neuronToTrain[i].activationFunction.GetSquashFunction(output) * error;
                double[] inputs = neuronToTrain[i].GetInputs();
                for (int j = 0; j < neuronToTrain[i].weights.Length; j++)
                {
                    //Modify each weight             
                    double changeDelta = (hiddenError * inputs[j]) * LEARN_RATE + neuronToTrain[i].previousChangeDelta[j] * MOMENTUM;
                    //Console.WriteLine("Change Delta: " + changeDelta + " final: " + neuronToTrain[i].weights[j] + changeDelta);
                    neuronToTrain[i].weights[j] += changeDelta;
                    neuronToTrain[i].previousChangeDelta[j] = changeDelta;

                    if (neuronToTrain[i].neuronInputs != null)
                    {
                        //Recurse on neuron inputs to correct weights
                        TrainHiddenLayer(error * (neuronToTrain[i].weights[j] - changeDelta), new Neuron[] { neuronToTrain[i].neuronInputs[j] });
                    }
                }
                //Modify Bias
                ModifyBias(neuronToTrain[i], hiddenError);
            }
        }
        [Obsolete]
        public void Train(double[] targets, Neuron[] neuronToTrain)
        {
            for (int i = 0; i < neuronToTrain.Length; i++)
            {
                double output = neuronToTrain[i].GetOutput();
                double error = neuronToTrain[i].activationFunction.GetSquashFunction(output) * (targets[i] - output);
                double[] inputs = neuronToTrain[i].GetInputs();

                for (int j = 0; j < neuronToTrain[i].weights.Length; j++)
                {
                    double changeDelta = (error * inputs[j]) * LEARN_RATE + neuronToTrain[i].previousChangeDelta[j] * MOMENTUM;

                    //Modify each weight
                    neuronToTrain[i].weights[j] += changeDelta;
                    neuronToTrain[i].previousChangeDelta[j] = changeDelta;
                    if (neuronToTrain[i].neuronInputs != null && error != 0)
                    {
                        TrainHiddenLayer(error, new Neuron[] { neuronToTrain[i].neuronInputs[j] });
                    }
                }
                //Modify Bias
                ModifyBias(neuronToTrain[i], error);
            }

        }



        public void ModifyBias(Neuron neuron, double error)
        {
            neuron.BIAS_WEIGHT += error * LEARN_RATE;
        }

        public double GetInput(Neuron neuron, int index)
        {
            //Index is the nth input for this neuron
            if (neuron.neuronInputs != null)
            {
                return neuron.neuronInputs[index].GetOutput();
            }
            else if (neuron.doubleInputs != null)
            {
                return neuron.doubleInputs[index];
            }
            throw new Exception("Both input and neuronInputs are null");
        }
    }
}

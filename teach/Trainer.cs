using NeuralNetwork.activationFunction;
using NeuralNetwork.neuron;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.teach
{
    public class Trainer
    {
        Network network;
        static double LEARN_RATE = 0.25;
        static double MOMENTUM = 0.01;
        public Trainer(Network network)
        {
            this.network = network;
        }

        public double Train(double[] inputs, double[] targets)
        {
            this.network.SetInputs(inputs);
            Neuron[] outputs = this.network.GetOutputNeurons();

            if (outputs.Length != targets.Length)
            {
                throw new Exception(String.Format("Output length mismatch {0} - {1}", outputs.Length, targets.Length));
            }

            Train(targets, outputs);

            double maxError = 0;

            double result = Math.Sqrt(targets[0] * targets[0] - outputs[0].GetOutput() * outputs[0].GetOutput());
            if (result > maxError)
            {
                maxError = result;
            }

            return Math.Round(Math.Abs(maxError), 4);
        }

        public void Train(double[] targets, Neuron[] neuronToTrain)
        {
            for (int i = 0; i < neuronToTrain.Length; i++)
            {
                double output = neuronToTrain[i].GetOutput();
                double error = neuronToTrain[i].activationFunction.GetSquashFunction(output) * (targets[i] - output);

                for (int j = 0; j < neuronToTrain[i].weights.Length; j++)
                {
                    double input = GetInput(neuronToTrain[i], j);
                    double changeDelta = (error * input) * LEARN_RATE + neuronToTrain[i].previousChangeDelta[j]*MOMENTUM;
                    neuronToTrain[i].previousChangeDelta[j] = changeDelta;
                    //Modify each weight
                    neuronToTrain[i].weights[j] += changeDelta;
                    if (neuronToTrain[i].neuronInputs != null)
                    {
                        //Recurse on neuron inputs to correct weights
                        TrainHiddenLayer(error, new Neuron[] { neuronToTrain[i].neuronInputs[j] });
                    }
                }
                //Modify Bias
                ModifyBias(neuronToTrain[i], error);
            }
        }

        public void TrainHiddenLayer(double error, Neuron[] neuronToTrain)
        {
            for (int i = 0; i < neuronToTrain.Length; i++)
            {
                double output = neuronToTrain[i].GetOutput();
                double hiddenError = neuronToTrain[i].activationFunction.GetSquashFunction(output) * error;
                for (int j = 0; j < neuronToTrain[i].weights.Length; j++)
                {
                    double input = GetInput(neuronToTrain[i], j);

                    //Modify each weight             
                    double changeDelta = (hiddenError * input) * LEARN_RATE;
                    neuronToTrain[i].weights[j] += changeDelta + neuronToTrain[i].previousChangeDelta[j] * MOMENTUM;
                    neuronToTrain[i].previousChangeDelta[j] = changeDelta;

                    if (neuronToTrain[i].neuronInputs != null)
                    {
                        //Recurse on neuron inputs to correct weights
                        TrainHiddenLayer(error, new Neuron[] { neuronToTrain[i].neuronInputs[j] });
                    }
                }
                //Modify Bias
                ModifyBias(neuronToTrain[i], hiddenError);
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

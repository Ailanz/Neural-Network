using NeuralNetwork.activationFunction;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.neuron
{
    public class Neuron
    {
        static Random rand = new Random();
        public double[] weights { get; set; }
        public double[] previousChangeDelta { get; set; }
        public double[] doubleInputs { get; set; }
        public Neuron[] neuronInputs { get; set; }

        public double BIAS = 1.0;
        public double BIAS_WEIGHT = 0.5;


        public ActivationFunction activationFunction = new Sigmoid();

        String ERROR_LENGTH_MISMATCH = "Weight length does not match input length, {0} and {1}";

        public Neuron(ActivationFunction activationFunction) 
        {
            this.activationFunction = activationFunction;
            BIAS_WEIGHT = rand.Next(-100, 100) / 100.0;
        }

        public Neuron(int numInputs, ActivationFunction activationFunction)
            : this(activationFunction)
        {
            RandomizeWeights(numInputs);
        }

        public Neuron(double[] weights, ActivationFunction activationFunction)
            : this(activationFunction)
        {
            this.weights = weights;
            this.previousChangeDelta = new double[weights.Length];
        }

        public Neuron(Neuron[] inputs, ActivationFunction activationFunction)
            : this(activationFunction)
        {
            this.neuronInputs = inputs;
            RandomizeWeights(this.neuronInputs.Length);
        }


        void RandomizeWeights(int length)
        {
            if (weights == null || weights.Length == 0)
            {
                weights = new double[length];
                previousChangeDelta = new double[length];
                for (int i = 0; i < length; i++)
                {
                    weights[i] = rand.Next(0, 100) / 100.0;
                    previousChangeDelta[i] = 0;
                }
            }
            else if (weights.Length != doubleInputs.Length)
            {
                throw new Exception(String.Format(ERROR_LENGTH_MISMATCH, weights.Length, doubleInputs.Length));
            }
        }

        public void SetActivationFunction(ActivationFunction activationFunction)
        {
            this.activationFunction = activationFunction;
        }
                
        public void SetWeights(double[] weights)
        {
            this.weights = weights;
        }

        public void SetNeuronInputs(Neuron[] inputs)
        {
            this.neuronInputs = inputs;
        }

        public double GetOutput()
        {
            double sum = 0;

            if (doubleInputs != null)
            {
                for (int i = 0; i < doubleInputs.Length; i++)
                {
                    sum += doubleInputs[i] * weights[i];
                }
            }
            else if (neuronInputs != null)
            {
                for (int i = 0; i < neuronInputs.Length; i++)
                {
                    sum += neuronInputs[i].GetOutput() * weights[i];
                }
            }
            sum += BIAS * BIAS_WEIGHT;
            return this.activationFunction.GetResult(sum);
        }  
    }
}

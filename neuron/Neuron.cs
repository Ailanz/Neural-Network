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

        public int weightsHashCode = 0;
        public double[] previousChangeDelta { get; set; }
        public double[] doubleInputs { get; set; }

        public double backPropogationError { get; set; }
        public Neuron[] neuronInputs { get; set; }

        public double cachedOutput = 0;

        public double BIAS = 1.0;
        public double BIAS_WEIGHT = 0.5;

        public ActivationFunction activationFunction = new Sigmoid();

        String ERROR_LENGTH_MISMATCH = "Weight length does not match input length, {0} and {1}";

        public Neuron(ActivationFunction activationFunction) 
        {
            this.activationFunction = activationFunction;
            backPropogationError = 0;
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

        public double[] GetInputs()
        {
            if(this.doubleInputs != null && this.doubleInputs.Length != 0)
            {
                return this.doubleInputs;
            }
            else if(this.neuronInputs != null && this.neuronInputs.Length != 0)
            {
                double[] inputs = new double[this.neuronInputs.Length];
                for(int i = 0; i < inputs.Length; i++)
                {
                    inputs[i] = this.neuronInputs[i].GetOutput();
                }
                return inputs;
            }
            throw new Exception("INPUTS ARE ALL NULL");
        }

        public double GetOutput()
        {
            double sum = 0;

            if (this.weights.GetHashCode() + this.BIAS_WEIGHT.GetHashCode() == this.weightsHashCode)
            {
                return this.cachedOutput;
            }

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

            if (!isInputLayer())
            {
                //Input Layer Does not use Bias
                sum += BIAS * BIAS_WEIGHT;
            }
            double output = this.activationFunction.GetResult(sum);
            this.cachedOutput = output;
            this.weightsHashCode = this.weights.GetHashCode() + BIAS_WEIGHT.GetHashCode();
            return output;
        }
  
        public bool isInputLayer()
        {
            if (this.doubleInputs == null || this.doubleInputs.Length == 0)
            {
                return false;
            }
            return true;
        }
    }
}

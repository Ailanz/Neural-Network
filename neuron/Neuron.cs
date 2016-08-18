using NeuralNetwork.activationFunction;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace NeuralNetwork.neuron
{
    [Serializable()]
    public class Neuron
    {

        public Network network { get; set; }

        static Random rand = new Random();
        public double[] weights { get; set; }

        public Boolean hasUpdated = true;
        double[] cachedInputs;

        public double[] previousChangeDelta { get; set; }
        public double[] doubleInputs { get; set; }

        [XmlIgnore]
        public double backPropogationError { get; set; }
        public Neuron[] neuronInputs { get; set; }

        [XmlIgnore]
        public double cachedOutput = 0;

        public double BIAS = 1.0;
        public double BIAS_WEIGHT = 0.5;

        public ActivationFunction activationFunction = Sigmoid.GetInstance();

        String ERROR_LENGTH_MISMATCH = "Weight length does not match input length, {0} and {1}";


        private Neuron(ActivationFunction activationFunction) 
        {
            this.activationFunction = activationFunction;
            backPropogationError = 0;
            BIAS_WEIGHT = rand.Next(-100, 100) / 100.0;
        }

        private Neuron(int numInputs, ActivationFunction activationFunction)
            : this(activationFunction)
        {
            RandomizeWeights(numInputs);
        }

        public Neuron(int numInputs, ActivationFunction activationFunction, Network network)
            : this(numInputs, activationFunction)
        {
            this.network = network;
        }

        public Neuron(Neuron[] inputs, ActivationFunction activationFunction, Network network)
            : this(activationFunction)
        {
            this.neuronInputs = inputs;
            this.network = network;
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
                    weights[i] = 0;// rand.Next(0, 100) / 100.0;
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

            if (!hasUpdated && this.cachedInputs == this.network.currentInput)
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
                    if (weights[i] > 1)
                    {
                        //weights[i] = 0.99999999999999;
                    }

                    if(weights[i] < -1)
                    {
                        //weights[i] = -0.99999999999999;
                    }
                    sum += neuronInputs[i].GetOutput() * weights[i];
                }
            }

            if (!isInputLayer())
            {
                //Input Layer Does not use Bias
                sum += BIAS * BIAS_WEIGHT;
            }
            double output = this.activationFunction.GetResult(sum);
            this.hasUpdated = false;
            this.cachedInputs = this.network.currentInput;
            this.cachedOutput = output;
            return output;
        }
  
        public bool isInputLayer()
        {
            return (this.doubleInputs == null || this.doubleInputs.Length == 0) ? false : true;
        }
    }
}

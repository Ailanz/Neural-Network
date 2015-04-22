using NeuralNetwork.activationFunction;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.neuron
{
    public class Network
    {
        Neuron[] inputLayerNeurons;
        Neuron[] hiddenLayerNeurons;
        Neuron[] secondHiddenLayerNeurons;
        Neuron[] outputLayerNeurons;

        public Network() { }

        //If no Activation Function is declared, it will use Sigmoid as default
        public Network(int numInputs, int numInputLayer, int numHiddenLayers,  int numSecondHiddenLayerNeurons, int numOutputs) 
            : this(numInputs, numInputLayer, numHiddenLayers, numSecondHiddenLayerNeurons, numOutputs, Sigmoid.GetInstance())
        {           
        }

        public Network(int numInputs, int numInputLayer, int numHiddenLayers, int numSecondHiddenLayerNeurons, int numOutputs, ActivationFunction activationFunction)
        {
            inputLayerNeurons = new Neuron[numInputLayer];
            hiddenLayerNeurons = new Neuron[numHiddenLayers];
            secondHiddenLayerNeurons = new Neuron[numSecondHiddenLayerNeurons];
            outputLayerNeurons = new Neuron[numOutputs];

            for (int i = 0; i < numInputLayer; i++)
            {
                inputLayerNeurons[i] = new Neuron(numInputs, activationFunction);
            }

            for (int i = 0; i < numHiddenLayers; i++)
            {
                hiddenLayerNeurons[i] = new Neuron(inputLayerNeurons, activationFunction);
            }

            for (int i = 0; i < numSecondHiddenLayerNeurons; i++)
            {
                secondHiddenLayerNeurons[i] = new Neuron(hiddenLayerNeurons, activationFunction);
            }

            for (int i = 0; i < numOutputs; i++)
            {
                if (secondHiddenLayerNeurons.Length != 0)
                {
                    outputLayerNeurons[i] = new Neuron(secondHiddenLayerNeurons, activationFunction);
                }
                else if (hiddenLayerNeurons.Length != 0)
                {
                    outputLayerNeurons[i] = new Neuron(hiddenLayerNeurons, activationFunction);
                }
                else
                {
                    outputLayerNeurons[i] = new Neuron(inputLayerNeurons, activationFunction);
                }
            }
            Console.WriteLine("Total number of neurons: " + (inputLayerNeurons.Length + hiddenLayerNeurons.Length + secondHiddenLayerNeurons.Length + outputLayerNeurons.Length));
        }



        public void SetInputLayerNeurons(Neuron[] neurons)
        {
            this.inputLayerNeurons = neurons;
        }

        public void SetHiddenLayerNeurons(Neuron[] neurons)
        {
            this.hiddenLayerNeurons = neurons;
        }

        public void SetOutputLayerNeurons(Neuron[] neurons)
        {
            this.outputLayerNeurons = neurons;
        }

       

        public void SetInputs(double[] inputs)
        {
            ResetAllCachedOutput();
            for (int i = 0; i < inputLayerNeurons.Length; i++)
            {
                inputLayerNeurons[i].doubleInputs = inputs;
            }
        }

        public void ResetAllCachedOutput()
        {
            for (int i = 0; i < inputLayerNeurons.Length; i++)
            {
                inputLayerNeurons[i].weightsHashCode = 0;
            }

            for (int i = 0; i < hiddenLayerNeurons.Length; i++)
            {
                hiddenLayerNeurons[i].weightsHashCode = 0;
            }

            for (int i = 0; i < secondHiddenLayerNeurons.Length; i++)
            {
                secondHiddenLayerNeurons[i].weightsHashCode = 0;
            }

            for (int i = 0; i < outputLayerNeurons.Length; i++)
            {
                outputLayerNeurons[i].weightsHashCode = 0;
            }
        }

        public Neuron[] GetOutputNeurons()
        {
            return outputLayerNeurons;
        }

        public double[] GetOutputsAsDoubleArray()
        {
            ResetAllCachedOutput();
            double[] outputs = new double[outputLayerNeurons.Length];
            for(int i=0; i < outputLayerNeurons.Length; i++)
            {
                outputs[i] = outputLayerNeurons[i].GetOutput();
            }
            return outputs;
        }

        public Neuron[] GetHiddenLayerNeurons()
        {
            return hiddenLayerNeurons;
        }

        public Neuron[] GetSecondHiddenLayerNeurons()
        {
            return secondHiddenLayerNeurons;
        }

        public Neuron[] GetInputNeurons()
        {
            return inputLayerNeurons;
        }

    }
}

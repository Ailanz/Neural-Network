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
        Neuron[] outputLayerNeurons;

        public Network() { }

        //If no Activation Function is declared, it will use Sigmoid as default
        public Network(int numInputs, int numInputLayer, int numHiddenLayers, int numOutputs) 
            : this(numInputs, numInputLayer, numHiddenLayers, numOutputs, Sigmoid.GetInstance())
        {           
        }

        public Network(int numInputs, int numInputLayer, int numHiddenLayers, int numOutputs, ActivationFunction activationFunction)
        {
            inputLayerNeurons = new Neuron[numInputLayer];
            hiddenLayerNeurons = new Neuron[numHiddenLayers];
            outputLayerNeurons = new Neuron[numOutputs];

            for (int i = 0; i < numInputLayer; i++)
            {
                inputLayerNeurons[i] = new Neuron(numInputs, activationFunction);
            }

            for (int i = 0; i < numHiddenLayers; i++)
            {
                hiddenLayerNeurons[i] = new Neuron(inputLayerNeurons, activationFunction);
            }

            for (int i = 0; i < numOutputs; i++)
            {
                outputLayerNeurons[i] = new Neuron(hiddenLayerNeurons, activationFunction);
            }
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
            for (int i = 0; i < inputLayerNeurons.Length; i++)
            {
                inputLayerNeurons[i].doubleInputs = inputs;
            }

        }

        public Neuron[] GetOutputNeurons()
        {
            return outputLayerNeurons;
        }

        public Neuron[] GetHiddenNeurons()
        {
            return hiddenLayerNeurons;
        }

        public Neuron[] GetInputNeurons()
        {
            return inputLayerNeurons;
        }

    }
}

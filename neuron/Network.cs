using NeuralNetwork.activationFunction;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace NeuralNetwork.neuron
{
    [Serializable()]
    public class Network
    {
        public Neuron[] inputLayerNeurons;
        public Neuron[] hiddenLayerNeurons;
        public Neuron[] secondHiddenLayerNeurons;
        public Neuron[] outputLayerNeurons;

        public Network() { }

        //If no Activation Function is declared, it will use Sigmoid as default
        public Network(int numInputLayer, int numHiddenLayers,  int numSecondHiddenLayerNeurons, int numOutputs) 
            : this(numInputLayer, numHiddenLayers, numSecondHiddenLayerNeurons, numOutputs, Sigmoid.GetInstance())
        {           
        }

        public Network(int numInputLayer, int numHiddenLayers, int numSecondHiddenLayerNeurons, int numOutputs, ActivationFunction activationFunction)
        {
            inputLayerNeurons = new Neuron[numInputLayer];
            hiddenLayerNeurons = new Neuron[numHiddenLayers];
            secondHiddenLayerNeurons = new Neuron[numSecondHiddenLayerNeurons];
            outputLayerNeurons = new Neuron[numOutputs];

            for (int i = 0; i < numInputLayer; i++)
            {
                inputLayerNeurons[i] = new Neuron(numInputLayer, activationFunction);
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
            Console.WriteLine("Number of connections {0} * {1} * {2} * {3}: " + (double)(inputLayerNeurons.Length * hiddenLayerNeurons.Length * secondHiddenLayerNeurons.Length * outputLayerNeurons.Length)
                , inputLayerNeurons.Length, hiddenLayerNeurons.Length , secondHiddenLayerNeurons.Length, outputLayerNeurons.Length);
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
                inputLayerNeurons[i].weightsHashCode = -1;
            }

            for (int i = 0; i < hiddenLayerNeurons.Length; i++)
            {
                hiddenLayerNeurons[i].weightsHashCode = -1;
            }

            for (int i = 0; i < secondHiddenLayerNeurons.Length; i++)
            {
                secondHiddenLayerNeurons[i].weightsHashCode = -1;
            }

            for (int i = 0; i < outputLayerNeurons.Length; i++)
            {
                outputLayerNeurons[i].weightsHashCode = -1;
            }
        }

        public Neuron[] GetOutputNeurons()
        {
            return outputLayerNeurons;
        }

        public double[] GetOutputsAsDoubleArray()
        {
            //ResetAllCachedOutput();
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

        public void SerializeNetworkToFile(string filepath)
        {
            //XmlSerializer SerializerObj = new XmlSerializer(typeof(Network), new Type[] { typeof(Neuron), typeof(Sigmoid), typeof(Tanh) });
            //// Create a new file stream to write the serialized object to a file
            //TextWriter WriteFileStream = new StreamWriter(filepath);
            //SerializerObj.Serialize(WriteFileStream, this);
            //// Cleanup
            //WriteFileStream.Close();
            FileStream fs = new FileStream(filepath, FileMode.Create);

            // Construct a BinaryFormatter and use it to serialize the data to the stream.
            BinaryFormatter formatter = new BinaryFormatter();
            try
            {
                formatter.Serialize(fs, this);
            }
            catch (SerializationException e)
            {
                Console.WriteLine("Failed to serialize. Reason: " + e.Message);
                throw;
            }
            finally
            {
                fs.Close();
            }
        }

        public static Network DeserializeNetworkFromFile(string filepath)
        {
            Network network = null;
            FileStream fs = new FileStream(filepath, FileMode.Open);
            try
            {
                BinaryFormatter formatter = new BinaryFormatter();

                // Deserialize the hashtable from the file and 
                // assign the reference to the local variable.
                network  = (Network)formatter.Deserialize(fs);
            }
            catch (SerializationException e)
            {
                Console.WriteLine("Failed to deserialize. Reason: " + e.Message);
                throw;
            }
            finally
            {
                fs.Close();
            }
            return network;
        }

    }
}

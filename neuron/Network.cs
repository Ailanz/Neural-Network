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
        public int numInputNeurons { get; set; }
        public int numOutputNeurons { get; set; }
        private List<int> numHiddenNeurons = new List<int>();

        public double[] currentInput;
        public Neuron[] inputLayerNeurons;
        public List<Neuron[]> hiddenLayerNeuronsList = new List<Neuron[]>();
        public Neuron[] outputLayerNeurons;
        ActivationFunction activationFunction;

        public Network(ActivationFunction activationFunction)
        {
            this.activationFunction = activationFunction;
        }

        public void ConstructNetwork()
        {
            if (numInputNeurons == 0 || numOutputNeurons == 0 || numHiddenNeurons.Count == 0)
            {
                throw new Exception("Invalid Network State, 1 or more layer of neurons are null");
            }

            inputLayerNeurons = new Neuron[numInputNeurons];
            outputLayerNeurons = new Neuron[numOutputNeurons];
            for (int i = 0; i < numInputNeurons; i++)
            {
                inputLayerNeurons[i] = new Neuron(numInputNeurons, activationFunction, this);
            }

            for (int i = 0; i < numHiddenNeurons.Count; i++)
            {
                Neuron[] hiddenenNeurons = new Neuron[numHiddenNeurons[i]];
                for (int j = 0; j < numHiddenNeurons[i]; j++)
                {
                    if (i == 0)
                    {
                        hiddenenNeurons[j] = new Neuron(inputLayerNeurons, activationFunction, this);
                    }
                    else
                    {
                        hiddenenNeurons[j] = new Neuron(hiddenLayerNeuronsList[i - 1], activationFunction, this);
                    }
                }
                hiddenLayerNeuronsList.Add(hiddenenNeurons);
            }

            for (int i = 0; i < numOutputNeurons; i++)
            {
                outputLayerNeurons[i] = new Neuron(hiddenLayerNeuronsList[hiddenLayerNeuronsList.Count - 1], activationFunction, this);
            }
        }


        public void SetNumberOfHiddenNeurons(params int[] numNeurons)
        {
            foreach (int i in numNeurons)
            {
                if(i==0)
                {
                    throw new Exception("Invalid neuron number: 0");
                }
                numHiddenNeurons.Add(i);
            }
        }

        public void SetInputs(double[] inputs)
        {
            this.currentInput = inputs;
            for (int i = 0; i < inputLayerNeurons.Length; i++)
            {
                inputLayerNeurons[i].doubleInputs = inputs;
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
            for (int i = 0; i < outputLayerNeurons.Length; i++)
            {
                outputs[i] = outputLayerNeurons[i].GetOutput();
            }
            return outputs;
        }


        public Neuron[] GetInputNeurons()
        {
            return inputLayerNeurons;
        }

        public void CalculateChangeDeltaByLayers()
        {
            double sumInputLayerChangeDelta = 0;
            foreach (Neuron n in GetInputNeurons())
            {
                sumInputLayerChangeDelta += n.previousChangeDelta.Sum(i => Math.Abs(i) / GetInputNeurons().Length);
            }

            List<double> hiddenDeltas = new List<double>();
            foreach (Neuron[] nl in hiddenLayerNeuronsList)
            {
                double sumHiddenLayerChangeDelta = 0;
                foreach (Neuron n in nl)
                {
                    sumHiddenLayerChangeDelta += n.previousChangeDelta.Sum(i => Math.Abs(i) / nl.Length);
                }
                hiddenDeltas.Add(sumHiddenLayerChangeDelta);
            }

            double sumHiddenDeltas = 0;
            foreach(double d in hiddenDeltas)
            {
                sumHiddenDeltas += d;
            }
            sumHiddenDeltas = sumHiddenDeltas;

            double sumOutputLayerChangeDelta = 0;
            foreach (Neuron n in GetOutputNeurons())
            {
                sumOutputLayerChangeDelta += n.previousChangeDelta.Sum(i => Math.Abs(i) / GetOutputNeurons().Length);
            }

            double total = sumInputLayerChangeDelta + sumHiddenDeltas + sumOutputLayerChangeDelta;

            String output = String.Format("Change Ratio: Input: {0}", Math.Round(sumInputLayerChangeDelta * 100 / total, 2));

            foreach (double d in hiddenDeltas)
            {
                output += String.Format(" Hidden: {0}, ", Math.Round(d * 100 / total, 2));
            }

            Console.WriteLine(output + " Output: {0}", Math.Round(sumOutputLayerChangeDelta * 100 / total, 2));
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
                network = (Network)formatter.Deserialize(fs);
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

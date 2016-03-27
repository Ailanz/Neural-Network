using NeuralNetwork.neuron;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNetwork.teach
{
    class TrainWorker
    {
        private Neuron neuron;

        const double learnRate = 0.30;
        const double momentum = 0.05;
        double error = 0;
        static Random random = new Random();
        const int RANDOM_SAMPLE = 35;

        //thread counter and blockers
        public static int counter = 0;
        public static ManualResetEvent doneEvent;


        static readonly object _object = new object();

        public TrainWorker(Neuron neuron, double error)
        {
            this.neuron = neuron;
            this.error = error;
        }

        public void ThreadPoolCallback(Object threadContext)
        {
            Work(neuron);
            if (Interlocked.Decrement(ref counter) == 0)
            {
                doneEvent.Set();
            }

        }

        public void Work(Neuron neuron)
        {
            lock (neuron)
            {
                double output = neuron.GetOutput(); //0.14
                double error = neuron.activationFunction.GetSquashFunction(output) * this.error;
                neuron.backPropogationError = 0;
                double[] inputs = neuron.GetInputs(); //0.15
                List<KeyValuePair<Neuron, double>> backPropErrorList = new List<KeyValuePair<Neuron, double>>();

                for (int j = 0; j < neuron.weights.Length; j++)
                {

                    double changeDelta = (error * inputs[j]) * learnRate + neuron.previousChangeDelta[j] * momentum;
                    //Modify each weight
                    if (!neuron.isInputLayer())
                    {
                        double backPropError = error * neuron.weights[j];

                        neuron.neuronInputs[j].backPropogationError += backPropError;

                    }
                    neuron.weights[j] += changeDelta;
                    neuron.previousChangeDelta[j] = changeDelta;
                    //Propogate the error back to previous layer neuron
                }
                //Modify Bias
                ModifyBias(neuron, error);
                neuron.hasUpdated = true;
            }
        }

        public void ModifyBias(Neuron neuron, double error)
        {
            neuron.BIAS_WEIGHT += error * learnRate;
        }
    }
}

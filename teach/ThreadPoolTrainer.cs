using NeuralNetwork.neuron;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNetwork.teach
{
    class ThreadPoolTrainer
    {
        Network network;
        static Random random = new Random();
        const int RANDOM_SAMPLE = 35;
        public static int counter = 0;

        public delegate void Callback(Network network, List<double[]> inputs, List<double[]> targets, double errorRate, int repetition);


        public ThreadPoolTrainer(Network network)
        {
            this.network = network;
        }

        public double Train(List<double[]> inputs, List<double[]> targets, double precision, Callback callback)
        {  
            double sumError = Double.MaxValue;
            int repetition = 0;
            while (sumError > precision)
            {
                sumError = 0;
                for (int i = 0; i < inputs.Count(); i++)
                {
                    sumError += TrainNetworkAndReturnErrorRate(inputs[i], targets[i]);
                }

                sumError = sumError / inputs.Count();

                if (callback != null)
                {
                    callback(this.network, inputs, targets, sumError, repetition);
                }

                repetition++;
            }
            Console.WriteLine("Stopped at: " + repetition + " with error: " + sumError );
            return sumError;
        }

        public double TrainNetworkAndReturnErrorRate(double[] inputs, double[] targets)
        {
            this.network.SetInputs(inputs);
            Neuron[] outputs = this.network.GetOutputNeurons();

            if (outputs.Length != targets.Length)
            {
                throw new Exception(String.Format("Output length mismatch {0} - {1}", outputs.Length, targets.Length));
            }

            //Input and Output layer neurons are always non-null
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

            return EstimateErrorRate(inputs, targets);           
        }


        public void TrainOutputLayer(double[] targets)
        {
            Neuron[] neuronsToTrain = this.network.GetOutputNeurons();
            ManualResetEvent _doneEvent = new ManualResetEvent(false);
            TrainWorker[] workers = new TrainWorker[neuronsToTrain.Length];
            TrainWorker.counter = neuronsToTrain.Length;
            TrainWorker.doneEvent = _doneEvent;

            for (int i = 0; i < neuronsToTrain.Length; i++)
            {
                workers[i] = new TrainWorker(neuronsToTrain[i], targets[i] - neuronsToTrain[i].GetOutput());
                ThreadPool.QueueUserWorkItem(workers[i].ThreadPoolCallback, i);
            }

            _doneEvent.WaitOne();
        }

        public void TrainLayerNeurons(Neuron[] neurons)
        {

            ManualResetEvent _doneEvent = new ManualResetEvent(false);
            TrainWorker[] workers = new TrainWorker[neurons.Length];
            TrainWorker.counter = neurons.Length;
            TrainWorker.doneEvent = _doneEvent; 
            counter = neurons.Length;
            for (int i = 0; i < neurons.Length; i++)
            {
                workers[i] = new TrainWorker(neurons[i], neurons[i].backPropogationError);
                ThreadPool.QueueUserWorkItem(workers[i].ThreadPoolCallback, i);
            }
            _doneEvent.WaitOne();
        }


        public double EstimateErrorRate(double[] inputs, double[] targets)
        {
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

    }
}

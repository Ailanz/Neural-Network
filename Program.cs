using NeuralNetwork.activationFunction;
using NeuralNetwork.neuron;
using NeuralNetwork.teach;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {

            //TestSimpleNetwork();
            TestBasicMath();


            Console.Read();
        }


        static void TestBasicNeurons()
        {
            //Neuron a1 = new Neuron(new double[] { 0.35, 0.9 });
            //Neuron a2 = new Neuron(new double[] { 0.35, 0.9 });
            //Neuron a3 = new Neuron(new Neuron[] { a1, a2 });
            //a1.SetWeights(new double[] { 0.1, 0.8 });
            //a2.SetWeights(new double[] { 0.4, 0.6 });
            //a3.SetWeights(new double[] { 0.3, 0.9 });
            //Console.WriteLine("Output: " + a1.GetOutput() + " | " + a2.GetOutput());
            //Console.WriteLine("Final Output: " + a3.GetOutput());
        }

        static void TestBasicMath()
        {
            Network network = new Network(1, 4, 8, 2, Sigmoid.GetInstance());
            Neuron output1 = network.GetOutputNeurons()[0];
            Neuron output2 = network.GetOutputNeurons()[1];
            Neuron output3 = network.GetOutputNeurons()[0];
            Console.WriteLine("Output: " + output1.GetOutput());
            Trainer trainer = new Trainer(network);
            // Console.WriteLine("Before Training: " + output1.GetOutput());
            for (int i = 0; i < 30000000; i++)
            {
                //FIX: training causes weights to go to infinity
                //trainer.Train(new double[] { 0.2, 0.1 }, new double[] { 0.3, 0.1, 0.02 });
                //trainer.Train(new double[] { 0.3, 0.2 }, new double[] { 0.5, 0.1, 0.06 });
                //trainer.Train(new double[] { 0.5, 0.3 }, new double[] { 0.8, 0.2, 0.15 });
                //trainer.Train(new double[] { 0.4, 0.2 }, new double[] { 0.6, 0.2 });

                //trainer.Train(new double[] { 0.4, 0.2 }, new double[] { 0.6, 0.2 });
                //trainer.Train(new double[] { 0.4, 0.3 }, new double[] { 0.7, 0.1 });
                //trainer.Train(new double[] { 0.7, 0.2 }, new double[] { 0.9, 0.5 });
                //trainer.Train(new double[] { 0.3, 0.1 }, new double[] { 0.4, 0.2 });
                //trainer.Train(new double[] { 0.53, 0.21 }, new double[] { 0.74, 0.32 });

                //trainer.Train(new double[] { 0.2, 0.1 }, new double[] { 0.3 });
                //trainer.Train(new double[] { 0.2, 0.3 }, new double[] { 0.5 });
                //trainer.Train(new double[] { 0.5, 0.3 }, new double[] { 0.8 });
                //trainer.Train(new double[] { 0.4, 0.2 }, new double[] { 0.6 });
                //trainer.Train(new double[] { 0.1, 0.2 }, new double[] { 0.3 });
                double errorAverage = 0;
                errorAverage += trainer.Train(new double[] { 0.5 }, new double[] { 1, 0 });
                errorAverage += trainer.Train(new double[] { 0.4 }, new double[] { 1, 0 });
                errorAverage += trainer.Train(new double[] { 0.44 }, new double[] { 1, 0 });
                errorAverage += trainer.Train(new double[] { 0.1 }, new double[] { 1, 0 });
                errorAverage += trainer.Train(new double[] { -0.5 }, new double[] { 0, 1 });
                errorAverage += trainer.Train(new double[] { -0.4 }, new double[] { 0, 1 });
                errorAverage += trainer.Train(new double[] { -0.1 }, new double[] { 0, 1 });
                errorAverage += trainer.Train(new double[] { -0.14 }, new double[] { 0, 1 });
                if (i % 1000 == 0)
                {
                    network.SetInputs(new double[] { 0.52 });
                    Console.WriteLine("Error Average: " + errorAverage*100 / 8 + "%. Checking 0.5: " + Math.Round(output1.GetOutput(),4));
                }

                if(Math.Abs(errorAverage / 8) < 0.07 && i > 2)
                {
                    Console.WriteLine("Breaking on " + i + " : " + errorAverage * 100 / 8 + "%");
                    break;
                }

                //trainer.Train(new double[] { 0.5}, new double[] { 0.4794255386 });
                //trainer.Train(new double[] { 0.3 }, new double[] { 0.29552020666 });
                //trainer.Train(new double[] { 0.8 }, new double[] { 0.7173560909 });
                //trainer.Train(new double[] { 0.62 }, new double[] { 0.58103516053 });
            }
            //network.SetInputs(new double[] { 0, -2 });
            network.SetInputs(new double[] { 0.72 });
            PrintResult(output1, output2);
            network.SetInputs(new double[] { 0.61 });
            PrintResult(output1, output2);
            network.SetInputs(new double[] { -0.08 });
            PrintResult(output1, output2);
            network.SetInputs(new double[] { -0.8 });
            PrintResult(output1, output2);

            //network.SetInputs(new double[] { 90 });
            //Console.WriteLine("After Training: " + output1.GetOutput());
        }

        static void PrintResult(Neuron output1, Neuron output2)
        {
            Console.WriteLine("After Training: " + Math.Round(output1.GetOutput(),4) + " " + Math.Round(output2.GetOutput(),4));

        }

        static double ErrorRate(double[] expected, double[] received)
        {
            double sumErrorRate = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                sumErrorRate += received[i]/expected[i];
            }
            sumErrorRate = sumErrorRate / expected.Length;
            return sumErrorRate;
        }
    }
}

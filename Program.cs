using NeuralNetwork.activationFunction;
using NeuralNetwork.neuron;
using NeuralNetwork.normalizer;
using NeuralNetwork.teach;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace NeuralNetwork
{
    class Program
    {

        static int width = 20;
        static double curErrorRate = 1;
        static void Main(string[] args)
        {

            //TestSimpleNetwork();
            //TestBasicMath();
            TestAdd();
            //ResizeImage();
            //TestImage(width);
            //TestImageFromSavedObj(width, @"D:\test.dat");
            //TestRecreateImage();
            //double[] normalize = new double[] { 2, 5, 7, -1 };
            //normalize = Normalizer.Normalize(normalize, -1, 7);
            //normalize = Normalizer.Denormalize(normalize, -1, 7);
            //Console.WriteLine(Sigmoid.GetInstance(5).GetResult(40));

            // Create a new XmlSerializer instance with the type of the test class            
 

            Console.Read();
        }

        public static void TestAdd()
        {
            Network network = new Network(Sigmoid.GetInstance());
            network.numInputNeurons = 2;
            network.numOutputNeurons = 1;
            network.SetNumberOfHiddenNeurons(2);
            network.ConstructNetwork();
            Neuron output1 = network.GetOutputNeurons()[0];
            Console.WriteLine("Output: " + output1.GetOutput());

            //Trainer trainer = new Trainer(network);
            //ThreadPoolTrainer trainer = new ThreadPoolTrainer(network);
            GpuTrainer trainer = new GpuTrainer(network);
             
            Console.WriteLine("Before Training: " + output1.GetOutput());

            List<double[]> inputTrainingList = new List<double[]>();
            inputTrainingList.Add(Normalizer.Normalize(new double[] { 1, 1 }, 0, 20));
            inputTrainingList.Add(Normalizer.Normalize(new double[] { 2, 3 }, 0, 20));
            inputTrainingList.Add(Normalizer.Normalize(new double[] { 3, 6 }, 0, 20));
            

            List<double[]> outputTrainingList = new List<double[]>();
            //outputTrainingList.Add(Normalizer.Normalize(new double[] { 2, 2, 2, 2, 2 }, 0, 20));
            //outputTrainingList.Add(Normalizer.Normalize(new double[] { 5, 5 ,5 ,5 ,5 }, 0, 20));
            //outputTrainingList.Add(Normalizer.Normalize(new double[] { 9, 9, 9, 9, 9 }, 0, 20));
            outputTrainingList.Add(Normalizer.Normalize(new double[] { 2}, 0, 20));
            outputTrainingList.Add(Normalizer.Normalize(new double[] { 5}, 0, 20));
            outputTrainingList.Add(Normalizer.Normalize(new double[] { 9}, 0, 20));

            //NeuralNetwork.teach.Trainer.Callback handler = CallbackMethod;
            //NeuralNetwork.teach.ThreadPoolTrainer.Callback handler = CallbackMethod;
            NeuralNetwork.teach.GpuTrainer.Callback handler = CallbackMethod;

            trainer.Train(inputTrainingList, outputTrainingList, 0.012, handler);
            network.SetInputs(Normalizer.Normalize(new double[] { 1, 1 }, 0, 20));
            PrintResult(network.GetOutputsAsDoubleArray());
            network.SetInputs(Normalizer.Normalize(new double[] { 2, 5 }, 0, 20));
            PrintResult(network.GetOutputsAsDoubleArray());
            network.SetInputs(Normalizer.Normalize(new double[] { 4, 8 }, 0, 20));
            PrintResult(network.GetOutputsAsDoubleArray());
        }

        static void PrintResult(double[] output)
        {
            Console.WriteLine("After Training: " + Normalizer.Denormalize(output,0,20)[0]);
        }

        public static void CallbackMethod(Network network, List<double[]> inputs, List<double[]> targets, double errorRate, int repetition)
        {
            //Stopwatch stopwatch = Stopwatch.StartNew();
            if (repetition % 1000 == 0)
            {
                Console.WriteLine("Error: " + errorRate);
                network.CalculateChangeDeltaByLayers();
                //Console.WriteLine("Current Error: " + Math.Round(errorRate, 5));
            }
            //if (curErrorRate - errorRate > 0.05)
            //{
            //    curErrorRate = errorRate;
            //    //CREATE IMAGE
            //    network.SetInputs(inputs[0]);
            //    double[] xyz = network.GetOutputsAsDoubleArray();
            //    double[] outputs1 = Normalizer.Denormalize(network.GetOutputsAsDoubleArray(), 0, 255);
            //    Bitmap createdImage = ImageHelper.DoubleArrayToBitmap(outputs1, width, width);
            //    createdImage.Save("C:\\Users\\Ailan\\Pictures\\test\\TEST" + repetition + ".png");

            //    network.SetInputs(inputs[1]);
            //    double[] abc = network.GetOutputsAsDoubleArray();
            //    double[] outputs2 = Normalizer.Denormalize(network.GetOutputsAsDoubleArray(), 0, 255);
            //    createdImage = ImageHelper.DoubleArrayToBitmap(outputs2, width, width);
            //    createdImage.Save("C:\\Users\\Ailan\\Pictures\\test\\TEST-" + repetition + ".png");

            //    Console.WriteLine("Error Rate: " + Math.Round(errorRate, 5));
            //        //Console.WriteLine("Error Rate: " + errorRate);
            //    stopwatch.Stop();
            //    Console.WriteLine(stopwatch.ElapsedMilliseconds);
            //    stopwatch = Stopwatch.StartNew();
            //}
        }

        static void TestImageFromSavedObj(int width, string filepath)
        {
            int height = width;
            Network network = Network.DeserializeNetworkFromFile(filepath);

            Image img1 = new Bitmap("C:\\Users\\Ailan\\Pictures\\folderImg.png");
            Image img2 = new Bitmap("C:\\Users\\Ailan\\Pictures\\folderImg2.png");
            Image img3 = new Bitmap("C:\\Users\\Ailan\\Pictures\\folderImg3.png");

            double[] imgArray1 = ImageHelper.ImageToDoubleArray(ImageHelper.ResizeImage(img1, width, height));
            double[] imgArray2 = ImageHelper.ImageToDoubleArray(ImageHelper.ResizeImage(img2, width, height));
            double[] imgArray3 = ImageHelper.ImageToDoubleArray(ImageHelper.ResizeImage(img3, width, height));

            network.SetInputs(Normalizer.Normalize(imgArray1, 0, 255));
            double[] outputs = Normalizer.Denormalize(network.GetOutputsAsDoubleArray(), 0, 255);
            Bitmap createdImage = ImageHelper.DoubleArrayToBitmap(outputs, width, height);
            createdImage.Save("C:\\Users\\Ailan\\Pictures\\test\\FinalImageFromSavedObj.png");

            network.SetInputs(Normalizer.Normalize(imgArray2, 0, 256));
            double[] outputs2 = Normalizer.Denormalize(network.GetOutputsAsDoubleArray(), 0, 255);
            createdImage = ImageHelper.DoubleArrayToBitmap(outputs2, width, height);
            createdImage.Save("C:\\Users\\Ailan\\Pictures\\test\\FinalImageFromSavedObj2.png");

            network.SetInputs(Normalizer.Normalize(imgArray3, 0, 256));
            double[] outputs3 = Normalizer.Denormalize(network.GetOutputsAsDoubleArray(), 0, 255);
            createdImage = ImageHelper.DoubleArrayToBitmap(outputs3, width, height);
            createdImage.Save("C:\\Users\\Ailan\\Pictures\\test\\FinalImageFromSavedObj3.png");

        }

        static void TestImage(int width)
        {
            int height = width;
            //Img1 = training set
            //Img2 = testing set
            Image img1 = new Bitmap("C:\\Users\\Ailan\\Pictures\\folderImg.png");
            Image img2 = new Bitmap("C:\\Users\\Ailan\\Pictures\\folderImg2.png");
            Image img3 = new Bitmap("C:\\Users\\Ailan\\Pictures\\folderImg3.png");


            Bitmap resized3R = ImageHelper.ResizeImage(img3, width, height);

            double[] imgArray1 = ImageHelper.ImageToDoubleArray(ImageHelper.ResizeImage(img1, width, height));
            double[] imgArray2 = ImageHelper.ImageToDoubleArray(ImageHelper.ResizeImage(img2, width, height));
            double[] imgArray3 = ImageHelper.ImageToDoubleArray(ImageHelper.ResizeImage(img3, width, height));


            int inputSize = imgArray1.Length;

            Network network = new Network(Sigmoid.GetInstance());
            network.numInputNeurons = inputSize;
            network.numOutputNeurons = width * height * 3;
            network.SetNumberOfHiddenNeurons(80,80);
            network.ConstructNetwork();
            //Network network = new Network(inputSize, inputSize / 3, inputSize / 3, width * height * 3, Sigmoid.GetInstance(1));
           // Network network = new Network(inputSize, 500, 500, width * height * 3, Sigmoid.GetInstance(1));

            Neuron output1 = network.GetOutputNeurons()[0];
            Console.WriteLine("Output: " + output1.GetOutput());

            ThreadPoolTrainer trainer = new ThreadPoolTrainer(network);
            //GpuTrainer trainer = new GpuTrainer(network);

            // Console.WriteLine("Before Training: " + output1.GetOutput());

            List<double[]> inputTrainingList = new List<double[]>();
            inputTrainingList.Add(Normalizer.Normalize(imgArray1, 0, 255));
            inputTrainingList.Add(Normalizer.Normalize(imgArray2, 0, 255));
            inputTrainingList.Add(Normalizer.Normalize(imgArray3, 0, 255));


            List<double[]> outputTrainingList = new List<double[]>();
            outputTrainingList.Add(Normalizer.Normalize(imgArray1, 0, 255));
            outputTrainingList.Add(Normalizer.Normalize(imgArray2, 0, 255));
            outputTrainingList.Add(Normalizer.Normalize(imgArray3, 0, 255));


            NeuralNetwork.teach.ThreadPoolTrainer.Callback handler = CallbackMethod;
            //NeuralNetwork.teach.GpuTrainer.Callback handler = CallbackMethod;

            trainer.Train(inputTrainingList, outputTrainingList, 0.10, handler);

             
            network.SetInputs(Normalizer.Normalize(imgArray1, 0, 256));
            double[] outputs = Normalizer.Denormalize(network.GetOutputsAsDoubleArray(), 0, 255);
            Bitmap createdImage = ImageHelper.DoubleArrayToBitmap(outputs, width, height);
            createdImage.Save("C:\\Users\\Ailan\\Pictures\\folderImgCREATED.png");

            network.SetInputs(Normalizer.Normalize(imgArray2, 0, 256));
            double[] outputs2 = Normalizer.Denormalize(network.GetOutputsAsDoubleArray(), 0, 255);
            createdImage = ImageHelper.DoubleArrayToBitmap(outputs2, width, height);
            createdImage.Save("C:\\Users\\Ailan\\Pictures\\folderImgCREATED2.png");

            network.SetInputs(Normalizer.Normalize(imgArray3, 0, 256));
            double[] outputs3 = Normalizer.Denormalize(network.GetOutputsAsDoubleArray(), 0, 255);
            createdImage = ImageHelper.DoubleArrayToBitmap(outputs3, width, height);
            createdImage.Save("C:\\Users\\Ailan\\Pictures\\folderImgCREATED3.png");
            Console.WriteLine("Equal: " + isEqual(outputs, outputs2) + ", " + isEqual(imgArray1, imgArray2));

            network.SerializeNetworkToFile(@"D:\test.dat");
        }

        public static bool isEqual(double[] a, double[] b)
        {
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] != b[i])
                {
                    return false;
                }
            }
            return true;
        }

    }
}

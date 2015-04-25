using NeuralNetwork.activationFunction;
using NeuralNetwork.neuron;
using NeuralNetwork.normalizer;
using NeuralNetwork.teach;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
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
            //TestBasicMath();
            //TestAdd();
            //ResizeImage();
            TestImage();
            //TestRecreateImage();
            double[] normalize = new double[] { 2, 5, 7, -1 };
            normalize = Normalizer.Normalize(normalize, -1, 7);
            normalize = Normalizer.Denormalize(normalize, -1, 7);
            //Console.WriteLine(Sigmoid.GetInstance(5).GetResult(40));
            foreach (double d in normalize)
            {
                //Console.WriteLine(d);
            }


            Console.Read();
        }

        public static void CallbackMethod(Network network, List<double[]> inputs, List<double[]> targets, int repetition)
        {
            //CREATE IMAGE
            network.SetInputs(inputs[0]);
            int width = 25;
            double[] xyz = network.GetOutputsAsDoubleArray();
            double[] outputs1 = Normalizer.Denormalize(network.GetOutputsAsDoubleArray(), 0, 255);
            Bitmap createdImage = ImageHelper.DoubleArrayToBitmap(outputs1, width, width);
            createdImage.Save("C:\\Users\\Ailan\\Pictures\\test\\TEST" + repetition + ".png");

            network.SetInputs(inputs[1]);
            double[] abc = network.GetOutputsAsDoubleArray();
            double[] outputs2 = Normalizer.Denormalize(network.GetOutputsAsDoubleArray(), 0, 255);
            createdImage = ImageHelper.DoubleArrayToBitmap(outputs2, width, width);
            createdImage.Save("C:\\Users\\Ailan\\Pictures\\test\\TEST-" + repetition + ".png");
        }

        static void ResizeImage()
        {
            int width = 5;
            int height = 5;
            Image img1 = new Bitmap("C:\\Users\\Ailan\\Pictures\\folderImg.png");
            Image img2 = new Bitmap("C:\\Users\\Ailan\\Pictures\\folderImg2.png");
            Bitmap resized1 = ImageHelper.ResizeImage(img1, width, height);
            Bitmap resized2 = ImageHelper.ResizeImage(img2, width, height);
            resized1.Save("C:\\Users\\Ailan\\Pictures\\folderImgNew.png");
            resized2.Save("C:\\Users\\Ailan\\Pictures\\folderImgNew2.png");
            //20x20 = 1654
            double[] imgArray1 = ImageHelper.ImageToDoubleArray(resized1);
            double[] imgArray2 = ImageHelper.ImageToDoubleArray(resized2);
        }

        static void TestImage()
        {
            int width = 25;
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

            Network network = new Network(inputSize, 2 * inputSize / 3, inputSize / 3, width * height * 3, Sigmoid.GetInstance(1));
            Neuron output1 = network.GetOutputNeurons()[0];
            Console.WriteLine("Output: " + output1.GetOutput());
            Trainer trainer = new Trainer(network);
            // Console.WriteLine("Before Training: " + output1.GetOutput());

            List<double[]> inputTrainingList = new List<double[]>();
            inputTrainingList.Add(Normalizer.Normalize(imgArray1, 0, 255));
            inputTrainingList.Add(Normalizer.Normalize(imgArray2, 0, 255));
            inputTrainingList.Add(Normalizer.Normalize(imgArray3, 0, 255));


            List<double[]> outputTrainingList = new List<double[]>();
            outputTrainingList.Add(Normalizer.Normalize(imgArray1, 0, 255));
            outputTrainingList.Add(Normalizer.Normalize(imgArray2, 0, 255));
            outputTrainingList.Add(Normalizer.Normalize(imgArray3, 0, 255));


            NeuralNetwork.teach.Trainer.Callback handler = CallbackMethod;
            trainer.Train(inputTrainingList, outputTrainingList, 0.05, handler);


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

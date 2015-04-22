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
            //Console.WriteLine(Directory.GetCurrentDirectory());
            foreach (double d in normalize)
            {
                //Console.WriteLine(d);
            }


            Console.Read();
        }

        static void TestRecreateImage()
        {
            int width = 20;
            int height = 20;
            Image img1 = new Bitmap("C:\\Users\\Ailan\\Pictures\\folderImg.png");
            Bitmap resized1 = ImageHelper.ResizeImage(img1, width, height);
            double[] image = ImageHelper.ImageToDoubleArray(resized1);
            Bitmap imgFromArray = ImageHelper.DoubleArrayToBitmap(image, width, height);
            imgFromArray.Save("C:\\Users\\Ailan\\Pictures\\folderImgRECONSTRUCT.png");
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
            int height = 25;
            //Img1 = training set
            //Img2 = testing set
            Image img1 = new Bitmap("C:\\Users\\Ailan\\Pictures\\folderImg.png");
            Image img1R = new Bitmap("C:\\Users\\Ailan\\Pictures\\folderImgNew.png");
            Image img2 = new Bitmap("C:\\Users\\Ailan\\Pictures\\folderImg2.png");
            Image img2R = new Bitmap("C:\\Users\\Ailan\\Pictures\\folderImgNew2.png");

            Bitmap resized1R = ImageHelper.ResizeImage(img1R, width, height);

            Bitmap resized2R = ImageHelper.ResizeImage(img2R, width, height);

            double[] imgArray1 = ImageHelper.ImageToDoubleArray(ImageHelper.ResizeImage(img1, width, height));
            double[] imgArray1R = ImageHelper.ImageToDoubleArray(resized1R);

            double[] imgArray2 = ImageHelper.ImageToDoubleArray(ImageHelper.ResizeImage(img2, width, height));
            double[] imgArray2R = ImageHelper.ImageToDoubleArray(resized2R);

            int inputSize = imgArray1.Length;

            Network network = new Network(inputSize, inputSize, 2, 0, width * height * 3, Sigmoid.GetInstance());
            Neuron output1 = network.GetOutputNeurons()[0];
            Console.WriteLine("Output: " + output1.GetOutput());
            Trainer trainer = new Trainer(network);
            // Console.WriteLine("Before Training: " + output1.GetOutput());

            List<double[]> inputTrainingList = new List<double[]>();
            inputTrainingList.Add(Normalizer.Normalize(imgArray1, 0, 256));
            inputTrainingList.Add(Normalizer.Normalize(imgArray2, 0, 256));

            List<double[]> outputTrainingList = new List<double[]>();
            outputTrainingList.Add(Normalizer.Normalize(imgArray1R, 0, 256));
            outputTrainingList.Add(Normalizer.Normalize(imgArray2R, 0, 256));

            trainer.Train(inputTrainingList, outputTrainingList, 0.20);


            network.SetInputs(Normalizer.Normalize(imgArray1, 0, 256));
            double[] outputs = Normalizer.Denormalize(network.GetOutputsAsDoubleArray(), 0, 256);
            Bitmap createdImage = ImageHelper.DoubleArrayToBitmap(outputs, width, height);
            createdImage.Save("C:\\Users\\Ailan\\Pictures\\folderImgCREATED.png");

            network.SetInputs(Normalizer.Normalize(imgArray2, 0, 256));
            double[] outputs2 = Normalizer.Denormalize(network.GetOutputsAsDoubleArray(), 0, 256);
            createdImage = ImageHelper.DoubleArrayToBitmap(outputs2, width, height);
            createdImage.Save("C:\\Users\\Ailan\\Pictures\\folderImgCREATED2.png");
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

        static void TestAdd()
        {
            Network network = new Network(2, 2, 3, 0, 1, Sigmoid.GetInstance());
            Neuron output1 = network.GetOutputNeurons()[0];
            Console.WriteLine("Output: " + output1.GetOutput());
            Trainer trainer = new Trainer(network);
            // Console.WriteLine("Before Training: " + output1.GetOutput());

            List<double[]> inputTrainingList = new List<double[]>();
            inputTrainingList.Add(Normalizer.Normalize(new double[] { 1, 1 }, 0, 100));
            inputTrainingList.Add(Normalizer.Normalize(new double[] { 2, 3 }, 0, 100));
            inputTrainingList.Add(Normalizer.Normalize(new double[] { 3, 6 }, 0, 100));
            inputTrainingList.Add(Normalizer.Normalize(new double[] { 27, 52 }, 0, 100));
            inputTrainingList.Add(Normalizer.Normalize(new double[] { 13, 9 }, 0, 100));
            inputTrainingList.Add(Normalizer.Normalize(new double[] { 43, 19 }, 0, 100));

            List<double[]> outputTrainingList = new List<double[]>();
            outputTrainingList.Add(Normalizer.Normalize(new double[] { 2 }, 0, 100));
            outputTrainingList.Add(Normalizer.Normalize(new double[] { 5 }, 0, 100));
            outputTrainingList.Add(Normalizer.Normalize(new double[] { 9 }, 0, 100));
            outputTrainingList.Add(Normalizer.Normalize(new double[] { 79 }, 0, 100));
            outputTrainingList.Add(Normalizer.Normalize(new double[] { 22 }, 0, 100));
            outputTrainingList.Add(Normalizer.Normalize(new double[] { 62 }, 0, 100));

            trainer.Train(inputTrainingList, outputTrainingList, 0.013);
            network.SetInputs(Normalizer.Normalize(new double[] { 1, 1 }, 0, 100));
            PrintResult(output1);
            network.SetInputs(Normalizer.Normalize(new double[] { 2, 5 }, 0, 100));
            PrintResult(output1);
            network.SetInputs(Normalizer.Normalize(new double[] { 4, 8 }, 0, 100));
            PrintResult(output1);
            network.SetInputs(Normalizer.Normalize(new double[] { 26, 42 }, 0, 100));
            PrintResult(output1);
        }

        static void TestBasicMath()
        {
            Network network = new Network(1, 2, 3, 1, 1, Tanh.GetInstance());
            Neuron output1 = network.GetOutputNeurons()[0];
            Neuron output2 = network.GetOutputNeurons()[0];
            Neuron output3 = network.GetOutputNeurons()[0];
            Console.WriteLine("Output: " + output1.GetOutput());
            Trainer trainer = new Trainer(network);
            // Console.WriteLine("Before Training: " + output1.GetOutput());

            List<double[]> inputTrainingList = new List<double[]>();
            inputTrainingList.Add(new double[] { 0.5 });
            inputTrainingList.Add(new double[] { 0.3 });
            inputTrainingList.Add(new double[] { 0.8 });
            inputTrainingList.Add(new double[] { 0.62 });
            inputTrainingList.Add(new double[] { -0.12 });
            inputTrainingList.Add(new double[] { -0.63 });
            inputTrainingList.Add(new double[] { -0.9 });


            List<double[]> outputTrainingList = new List<double[]>();
            outputTrainingList.Add(new double[] { 0.4794255386 });
            outputTrainingList.Add(new double[] { 0.29552020666 });
            outputTrainingList.Add(new double[] { 0.7173560909 });
            outputTrainingList.Add(new double[] { 0.58103516053 });
            outputTrainingList.Add(new double[] { -0.11971220728 });
            outputTrainingList.Add(new double[] { -0.58914475794 });
            outputTrainingList.Add(new double[] { -0.78332690962 });


            trainer.Train(inputTrainingList, outputTrainingList, 0.0015);


            //network.SetInputs(new double[] { 0, -2 });
            network.SetInputs(new double[] { 0.72 }); //0.65938467197
            PrintResult(output1, output2);
            network.SetInputs(new double[] { 0.61 }); //0.5728674601
            PrintResult(output1, output2);
            network.SetInputs(new double[] { -0.08 }); //-0.0799991466
            PrintResult(output1, output2);
            network.SetInputs(new double[] { -0.8 }); //-0.799991466
            PrintResult(output1, output2);

            //network.SetInputs(new double[] { 90 });
            //Console.WriteLine("After Training: " + output1.GetOutput());
        }
        static void PrintResult(Neuron output1)
        {
            Console.WriteLine("After Training: " + Math.Round(output1.GetOutput() * 100, 4));
        }

        static void PrintResult(Neuron output1, Neuron output2)
        {
            Console.WriteLine("After Training: " + Math.Round(output1.GetOutput(), 4) + " " + Math.Round(output2.GetOutput(), 4));
        }

        static double ErrorRate(double[] expected, double[] received)
        {
            double sumErrorRate = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                sumErrorRate += received[i] / expected[i];
            }
            sumErrorRate = sumErrorRate / expected.Length;
            return sumErrorRate;
        }
    }
}

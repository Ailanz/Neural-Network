using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.normalizer
{
    public class Normalizer
    {
        public static double[] Normalize(double[] data, double min, double max)
        {
            if(data == null || data.Length==0 || min == max)
            {
                throw new Exception("Data is not initialized or Empty or Min == Max!");
            }

            double[] result = new double[data.Length];

            for(int i=0; i < data.Length; i++)
            {
                result[i] = (data[i] - min) / (max - min);
            }
            return result;
        }

        public static double[] Denormalize(double[] data, double min, double max)
        {
            double[] result = new double[data.Length];
            for(int i=0; i < data.Length; i++)
            {
                result[i] = data[i] * (max - min) + min;
            }
            return result;
        }
    }
}

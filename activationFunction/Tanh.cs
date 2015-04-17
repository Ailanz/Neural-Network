using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.activationFunction
{
    public class Tanh : ActivationFunction
    {
        public override double GetResult(double value)
        {
            return Math.Tanh(value);
        }

        public override double GetSquashFunction(double value)
        {
            return 1- value*value;
        }

        public static ActivationFunction GetInstance()
        {
            return new Tanh();
        }
    }
}

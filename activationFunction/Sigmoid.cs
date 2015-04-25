using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.activationFunction
{
    public class Sigmoid : ActivationFunction
    {
        double stretch = 1;

        public Sigmoid(double scale)
        {
            this.stretch = scale;
        }

        public override double GetResult(double value)
        {
            return 1 / (1 + Math.Exp(-value*stretch));
        }

        public override double GetSquashFunction(double value)
        {
            return value * (1 - value);
        }

        public static ActivationFunction GetInstance()
        {
            return GetInstance(1);
        }

        public static ActivationFunction GetInstance(double scale)
        {
            return new Sigmoid(scale);
        }
    }
}

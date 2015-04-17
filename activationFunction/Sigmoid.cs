using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.activationFunction
{
    public class Sigmoid : ActivationFunction
    {
        
        public override double GetResult(double value)
        {
            return 1 / (1 + Math.Exp(-value));
        }

        public override double GetSquashFunction(double value)
        {
            return value * (1.0 - value);
        }

        public static ActivationFunction GetInstance()
        {
            return new Sigmoid();
        }
    }
}

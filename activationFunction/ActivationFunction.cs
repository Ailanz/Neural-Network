using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.activationFunction
{
    public abstract class ActivationFunction
    {
        public abstract double GetResult(double value);
        public abstract double GetSquashFunction(double value);
    }
}

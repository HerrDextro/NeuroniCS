using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_Neuronics
{
    public class ActivationFunctions
    {
        
    
        public interface IActivationFunction
        {
            double Activate(double x);
            double Derivative(double x);
        }

        public class SigmoidActivation : IActivationFunction
        {
            public double Activate(double x)
            {
                return 1.0 / (1.0 + Math.Exp(-x));
            }

            public double Derivative(double x)
            {
                double sigmoid = Activate(x);
                return sigmoid * (1 - sigmoid);
            }
        }

        public class ReLUActivation : IActivationFunction
        {
            public double Activate(double x)
            {
                return Math.Max(0, x);
            }

            public double Derivative(double x)
            {
                return x > 0 ? 1 : 0;
            }
        }
    }
}


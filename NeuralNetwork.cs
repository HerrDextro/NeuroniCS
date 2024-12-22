using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TicTacToe_Neuronics.ActivationFunctions;

namespace TicTacToe_Neuronics
{
    //9 input neurons, 2 layers of 12 hidden, 9 output neurons.
    //need neurons themselves, synapses and things to hold it all together

    //Bias, backpropagation and input normalization are all recommendations and code from Claude AI
    //change output layer to SoftMax function recommended by Claude for changing binary classification into probabilities

    public class NeuralNetwork
    {
        public List<Neuron> InputLayer { get; set; }
        public List<Neuron> HiddenLayer1 { get; set; }
        public List<Neuron> HiddenLayer2 { get; set; }
        public List<Neuron> OutputLayer { get; set; }

        public NeuralNetwork(int inputNeurons, int hiddenNeurons1, int hiddenNeurons2, int outputNeurons)
        {
            InputLayer = new List<Neuron>();
            HiddenLayer1 = new List<Neuron>();
            HiddenLayer2 = new List<Neuron>();
            OutputLayer = new List<Neuron>();

            for (int i = 0; i < inputNeurons; i++)
                InputLayer.Add(new Neuron());

            for (int i = 0; i < hiddenNeurons1; i++)
                HiddenLayer1.Add(new Neuron());

            for (int i = 0; i < hiddenNeurons2; i++)
                HiddenLayer2.Add(new Neuron());

            for (int i = 0; i < outputNeurons; i++)
                OutputLayer.Add(new Neuron());

            ConnectLayers(InputLayer, HiddenLayer1);
            ConnectLayers(HiddenLayer1, HiddenLayer2);
            ConnectLayers(HiddenLayer2, OutputLayer);
        }

        private void ConnectLayers(List<Neuron> layer1, List<Neuron> layer2)
        {
            Random rand = new Random();
            foreach (var neuron1 in layer1)
            {
                foreach (var neuron2 in layer2)
                {
                    var synapse = new Synapse(neuron1, neuron2, rand.NextDouble());
                    neuron1.OutputSynapses.Add(synapse);
                    neuron2.InputSynapses.Add(synapse);
                }
            }
        }

        public void FeedForward(double[] inputCoords)
        {
            for (int i = 0; i < inputCoords.Length; i++)
            {
                InputLayer[i].Output = inputCoords[i];
            }

            foreach (var neuron in HiddenLayer1)
            {
                neuron.CalculateOutput();
            }
            foreach (var neuron in HiddenLayer2)
            {
                neuron.CalculateOutput();
            }

            // Calculate raw inputs for output layer
            foreach (var neuron in OutputLayer)
            {
                neuron.CalculateOutput();
            }

            // Apply softmax to output layer
            var softmaxLayer = new SoftmaxLayer(OutputLayer);
            softmaxLayer.ApplySoftmax();
        }

        public void Train(double[] inputs, double[] targetOutputs, double learningRate = 0.1)
        {
            // Forward pass
            FeedForward(inputs);

            // Calculate output layer deltas
            for (int i = 0; i < OutputLayer.Count; i++)
            {
                var neuron = OutputLayer[i];
                var output = neuron.Output;
                var target = targetOutputs[i];
                neuron.Delta = output * (1 - output) * (target - output);
            }

            // Calculate hidden layer deltas
            void CalculateLayerDeltas(List<Neuron> layer)
            {
                foreach (var neuron in layer)
                {
                    double sum = neuron.OutputSynapses.Sum(s => s.Weight * s.OutputNeuron.Delta);
                    neuron.Delta = neuron.Output * (1 - neuron.Output) * sum;
                }
            }

            CalculateLayerDeltas(HiddenLayer2);
            CalculateLayerDeltas(HiddenLayer1);

            // Update weights
            void UpdateLayer(List<Neuron> layer)
            {
                foreach (var neuron in layer)
                {
                    foreach (var synapse in neuron.OutputSynapses)
                    {
                        synapse.Weight += learningRate * synapse.InputNeuron.Output * synapse.OutputNeuron.Delta;
                    }
                }
            }

            UpdateLayer(HiddenLayer1);
            UpdateLayer(HiddenLayer2);
            UpdateLayer(InputLayer);
        }

        public void SaveWeights(string profileName)
        {
            string filename = $"profile_{profileName}.txt";
            using (StreamWriter writer = new StreamWriter(filename))
            {
                // Save weights for each layer
                SaveLayerWeights(InputLayer, writer, "Input");
                SaveLayerWeights(HiddenLayer1, writer, "Hidden1");
                SaveLayerWeights(HiddenLayer2, writer, "Hidden2");
                SaveLayerWeights(OutputLayer, writer, "Output");
            }
        }

        private void SaveLayerWeights(List<Neuron> layer, StreamWriter writer, string layerName)
        {
            writer.WriteLine($"[{layerName}Layer]");
            foreach (var neuron in layer)
            {
                foreach (var synapse in neuron.OutputSynapses)
                {
                    writer.WriteLine($"{synapse.Weight}");
                }
            }
        }

        public void LoadWeights(string profileName)
        {
            string filename = $"profile_{profileName}.txt";
            if (!File.Exists(filename))
            {
                Console.WriteLine($"Profile {profileName} not found!");
                return;
            }

            using (StreamReader reader = new StreamReader(filename))
            {
                LoadLayerWeights(InputLayer, reader);
                LoadLayerWeights(HiddenLayer1, reader);
                LoadLayerWeights(HiddenLayer2, reader);
                LoadLayerWeights(OutputLayer, reader);
            }
        }

        private void LoadLayerWeights(List<Neuron> layer, StreamReader reader)
        {
            string header = reader.ReadLine(); // Read layer header
            foreach (var neuron in layer)
            {
                foreach (var synapse in neuron.OutputSynapses)
                {
                    if (double.TryParse(reader.ReadLine(), out double weight))
                    {
                        synapse.Weight = weight;
                    }
                }
            }
        }
    }



    public class Synapse
    {
        public Neuron InputNeuron { get; set; }
        public Neuron OutputNeuron { get; set; }
        public double Weight { get; set; }

        public Synapse(Neuron inputNeuron, Neuron outputNeuron, double weight)
        {
            InputNeuron = inputNeuron;
            OutputNeuron = outputNeuron;
            Weight = weight;
        }
    }


    public class Neuron
    {
        public List<Synapse> InputSynapses { get; set; }
        public List<Synapse> OutputSynapses { get; set; }
        public double Output { get; set; }
        public double InputSum { get; private set; }
        public double Delta { get; set; } //for backpropagations
        public IActivationFunction ActivationFunction { get; set; }

        public Neuron(IActivationFunction activation = null)
        {
            InputSynapses = new List<Synapse>();
            OutputSynapses = new List<Synapse>();
            ActivationFunction = activation ?? new SigmoidActivation();
        }

        public void CalculateOutput()
        {
            InputSum = 0;
            foreach (var synapse in InputSynapses)
            {
                InputSum += synapse.Weight * synapse.InputNeuron.Output;
            }
            Output = ActivationFunction.Activate(InputSum);
        }
    }

    public class SoftmaxLayer
    {
        private List<Neuron> neurons;

        public SoftmaxLayer(List<Neuron> neurons)
        {
            this.neurons = neurons;
        }

        public void ApplySoftmax()
        {
            double max = neurons.Max(n => n.InputSum);
            double sum = 0;

            foreach (var neuron in neurons)
            {
                neuron.Output = Math.Exp(neuron.InputSum - max);
                sum += neuron.Output;
            }

            foreach (var neuron in neurons)
            {
                neuron.Output /= sum;
            }
        }
    }
}

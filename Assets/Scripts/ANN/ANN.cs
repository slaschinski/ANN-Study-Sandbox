using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ANN {

    private double alpha = 0.8;
    private int numInputs;
    List<Layer> layers = new List<Layer>();

	public ANN(int numberOfInputs)
    {
        this.numInputs = numberOfInputs;
    }

    public void AddLayer(int numberOfNeurons, string activationFunction)
    {
        int numberOfInputs;
        if (layers.Count == 0)
        {
            numberOfInputs = this.numInputs;
        }
        else
        {
            numberOfInputs = layers[layers.Count - 1].neurons.Count;
        }
        layers.Add(new Layer(numberOfNeurons, numberOfInputs, activationFunction));
    }


    public List<double> Predict(List<double> inputValues)
    {
        List<double> inputs;
        List<double> outputs = new List<double>();

        if (inputValues.Count != numInputs)
        {
            Debug.Log("ERROR: Number of Inputs must be " + numInputs);
            return outputs;
        }

        for (int i = 0; i < layers.Count; i++)
        {
            if (i == 0)
            {
                inputs = new List<double>(inputValues);
            }
            else
            {
                inputs = new List<double>(outputs);
            }
            outputs.Clear();

            for (int j = 0; j < layers[i].neurons.Count; j++)
            {
                double dotProduct = 0;
                layers[i].neurons[j].inputs.Clear();

                for (int k = 0; k < layers[i].neurons[j].weights.Count; k++)
                {
                    layers[i].neurons[j].inputs.Add(inputs[k]);
                    dotProduct += layers[i].neurons[j].weights[k] * inputs[k];
                }

                dotProduct += layers[i].neurons[j].bias;
                layers[i].neurons[j].output = ActivationFunction(dotProduct, layers[i].activationFunction);
                outputs.Add(layers[i].neurons[j].output);
            }
        }

        return outputs;
    }

    public List<double> Train(List<double> inputValues, List<double> desiredOutputValues)
    {
        List<double> outputs = Predict(inputValues);

        UpdateWeights(outputs, desiredOutputValues);

        return outputs;
    }

    public void UpdateWeights(List<double> outputs, List<double> desiredOutputValues)
    {
        double error;

        for (int i = layers.Count - 1; i >= 0; i--)
        {
            for (int j = 0; j < layers[i].neurons.Count; j++)
            {
                if (i == (layers.Count - 1))
                {
                    error = desiredOutputValues[j] - outputs[j];
                    layers[i].neurons[j].errorGradient = outputs[j] * (1 - outputs[j]) * error;
                }
                else
                {
                    double errorGradSum = 0;
                    for (int p = 0; p < layers[i + 1].neurons.Count; p++)
                    {
                        errorGradSum += layers[i + 1].neurons[p].errorGradient * layers[i + 1].neurons[p].weights[j];
                    }
                    layers[i].neurons[j].errorGradient = layers[i].neurons[j].output * (1 - layers[i].neurons[j].output) * errorGradSum;
                }
                for(int k = 0; k < layers[i].neurons[j].inputs.Count; k++)
                {
                    if (i == (layers.Count - 1))
                    {
                        error = desiredOutputValues[j] - outputs[j];
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * error;
                    }
                    else
                    {
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * layers[i].neurons[j].errorGradient;
                    }
                }
                
                layers[i].neurons[j].bias += alpha * layers[i].neurons[j].errorGradient;
            }
        }
    }

    private double ActivationFunction(double dotProduct, string activationFunction)
    {
        switch (activationFunction)
        {
            case "Linear":
                return Linear(dotProduct);
            case "ReLU":
                return ReLU(dotProduct);
            case "Sigmoid":
                return Sigmoid(dotProduct);
            case "Step":
                return Step(dotProduct);
            case "TanH":
                return TanH(dotProduct);
            default:
                Debug.LogError("Unknown activation function used!");
                return 0;
        }
    }

    private double Step(double value)
    {
        if (value < 0) return 0;
        else return 1;
    }

    private double Sigmoid(double value)
    {
        return 1.0f / (1.0f + (double)System.Math.Exp(-value));
    }

    double TanH(double value)
    {
        return 2.0f / (1.0f + (double)System.Math.Exp(-2.0f * value)) - 1.0f;
    }

    private double ReLU(double x)
    {
        return System.Math.Max(0, x); // x < 0 ? 0 : x;
    }

    private double Linear(double value)
    {
        return value;
    }
}

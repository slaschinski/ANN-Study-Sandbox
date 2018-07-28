using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Neuron
{
    public double bias;
    public string activationFunction;
    public double output;
    public double errorGradient;
    public List<double> inputs = new List<double>();
    public List<double> weights = new List<double>();

    // Initialize bias and weights with random values
    public Neuron(int numberOfInputs, string activationFunction = "Sigmoid")
    {
        for (int i = 0; i < numberOfInputs; i++)
        {
            weights.Add(UnityEngine.Random.Range(-1.0f, 1.0f));
        }
        bias = UnityEngine.Random.Range(-1.0f, 1.0f);

        this.activationFunction = activationFunction;
    }

    public void setInputs(List<double> inputs)
    {
        if (inputs.Count != weights.Count)
        {
            Debug.LogError("Number of inputs should be " + weights.Count);
            this.inputs = new List<double>();
        }
        this.inputs = new List<double>(inputs);
    }

    public double CalculateOutput(List<double> inputs)
    {

        if (inputs.Count == 0)
        {
            Debug.LogError("No inputs set");
            output = 0;
            return output;
        }

        double dotProduct = 0;

        // step through every input
        for (int i = 0; i < inputs.Count; i++)
        {
            // calculate the product and add it to the dot product
            dotProduct += weights[i] * inputs[i];
        }

        // add bias to the calculated dot product
        dotProduct += bias;
        // use activate function depending on settings to calculate the output
        output = ActivationFunction(dotProduct, activationFunction);
        return output;
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

    private double TanH(double value)
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

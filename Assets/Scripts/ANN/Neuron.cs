using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Neuron
{
    public string activationFunction;
    private List<double> inputs = new List<double>();
    private List<double> weights = new List<double>();
    private List<double> deltas = new List<double>();
    private double bias;
    private double biasDelta;
    private double dotProduct;
    private double output;
    private double errorGradient;
    GameObject debugNeuron;

    // Initialize bias and weights with random values
    public Neuron(int numberOfInputs, int layerSize, string activationFunction = "Sigmoid", GameObject debugNeuron = null)
    {
        this.debugNeuron = debugNeuron;

        for (int i = 0; i < numberOfInputs; i++)
        {
            //weights.Add(Random.Range(-0.1f, 0.1f));
            //weights.Add(Random.Range(layerSize, numberOfInputs) * Mathf.Sqrt(2.0f / numberOfInputs));
            weights.Add(NextGaussian());
        }
        bias = Random.Range(-0.1f, 0.1f);

        if (debugNeuron != null)
        {
            debugNeuron.GetComponent<DebugNeuron>().setWeights(weights);
            debugNeuron.GetComponent<DebugNeuron>().setBiasColor(bias);
        }

        this.activationFunction = activationFunction;
    }

    public List<double> getInputs()
    {
        return inputs;
    }

    public double getInput(int i)
    {
        return inputs[i];
    }

    public void setInputs(List<double> inputs)
    {
        if (inputs.Count != weights.Count)
        {
            Debug.LogError("Number of inputs should be " + weights.Count);
            this.inputs = new List<double>();
        }
        this.inputs = new List<double>(inputs);

        if (debugNeuron != null)
        {
            debugNeuron.GetComponent<DebugNeuron>().setWeightedInputs(weights, inputs);
        }
    }

    public List<double> getWeights()
    {
        return weights;
    }

    public double getWeight(int i)
    {
        return weights[i];
    }

    public int getWeightCount()
    {
        return weights.Count;
    }

    public void setWeights(List<double> newWeights)
    {
        if (newWeights.Count != weights.Count)
        {
            Debug.LogError("Number of weights should be " + weights.Count);
        }
        weights = new List<double>(newWeights);

        if (debugNeuron != null)
        {
            debugNeuron.GetComponent<DebugNeuron>().setWeights(weights);
        }
    }

    public List<double> getDeltas()
    {
        return deltas;
    }

    public void setDeltas(List<double> newDeltas)
    {
        if (newDeltas.Count != weights.Count)
        {
            Debug.LogError("Number of deltas should be " + weights.Count);
        }
        deltas = new List<double>(newDeltas);
    }

    public void addToDeltas(List<double> newDeltas)
    {
        if (newDeltas.Count != weights.Count)
        {
            Debug.LogError("Number of deltas should be " + weights.Count);
        }
        for (int i = 0; i < newDeltas.Count; i++)
        {
            deltas[i] += newDeltas[i];
        }
        
    }

    public double getBias()
    {
        return bias;
    }

    public void setBias(double bias)
    {
        this.bias = bias;

        if (debugNeuron != null)
        {
            debugNeuron.GetComponent<DebugNeuron>().setBiasColor(bias);
        }
    }

    public void setBiasDelta(double biasDelta)
    {
        this.biasDelta = biasDelta;
    }

    public void addToBiasDelta(double biasDelta)
    {
        this.biasDelta += biasDelta;
    }

    public double getDotProduct()
    {
        return dotProduct;
    }

    public double getOutput()
    {
        return output;
    }

    public double getErrorGradient()
    {
        return errorGradient;
    }

    public void setErrorGradient(double errorGradient)
    {
        this.errorGradient = errorGradient;
    }

    public double CalculateOutput()
    {

        if (inputs.Count == 0)
        {
            Debug.LogError("No inputs set");
            output = 0;
            return output;
        }

        // lets start with the bias which should be added to the dot product
        dotProduct = bias;

        // step through every input
        for (int i = 0; i < inputs.Count; i++)
        {
            // calculate the product and add it to the dot product
            dotProduct += weights[i] * inputs[i];
        }
        
        // use activate function depending on settings to calculate the output
        output = ActivationFunction(dotProduct);

        if (debugNeuron != null)
        {
            debugNeuron.GetComponent<DebugNeuron>().setOutput((float)output);
        }
        return output;
    }

    public double CalculateErrorGradient(double error)
    {
        errorGradient = ActivationDerivative(dotProduct) * error;
        if (errorGradient > 1.0d)
        {
            Debug.LogWarning("Gradient clipped to 1 - was at " + errorGradient);
            errorGradient = 1.0d;
        } else if (errorGradient < -1.0d)
        {
            Debug.LogWarning("Gradient clipped to -1 - was at " + errorGradient);
            errorGradient = -1.0d;
        }
        return errorGradient;
    }

    public void ApplyDeltas(double lambda = 0.0f)
    {
        if (lambda != 0.0f)
        {
            //regulations
            for (int i = 0; i < weights.Count; i++)
            {
                //weights[i] -= Mathf.Sign((float)weights[i]) * lambda;
                weights[i] -= Mathf.Sign((float)weights[i]) * lambda * weights[i] * weights[i];
                //weights[i] -= lambda * weights[i]; // L2 regularization
            }
            //bias -= Mathf.Sign((float)bias) * lambda;
            bias -= Mathf.Sign((float)bias) * lambda * bias * bias;
            //bias -= lambda * bias; // L2 regularization
        }

        List<double> newWeights = new List<double>();
        for (int i = 0; i < weights.Count; i++)
        {
            newWeights.Add(weights[i] += deltas[i]);
        }
        setWeights(newWeights);

        setBias(bias + biasDelta);
    }

    private double ActivationFunction(double dotProduct)
    {
        switch (activationFunction)
        {
            case "Linear":
                return dotProduct;
            case "ReLU":
                return ReLU(dotProduct);
            case "LeakyReLU":
                return LeakyReLU(dotProduct);
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

    private double ActivationDerivative(double dotProduct)
    {
        switch (activationFunction)
        {
            case "Linear":
                return 1; // is always 1
            case "ReLU":
                return ReLUDeri(dotProduct);
            case "LeakyReLU":
                return LeakyReLUDeri(dotProduct);
            case "Sigmoid":
                return SigmoidDeri(dotProduct);
            case "Step":
                return SigmoidDeri(dotProduct); // should work
            case "TanH":
                return TanHDeri(dotProduct);
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
        return 1.0d / (1.0d + System.Math.Exp(-value));
    }

    private double SigmoidDeri(double value)
    {
        return Sigmoid(value) * (1.0d - Sigmoid(value));
    }

    private double TanH(double value)
    {
        return 2.0d / (1.0d + System.Math.Exp(-2.0d * value)) - 1.0d;
    }

    private double TanHDeri(double value)
    {
        return 1 - System.Math.Exp(value);
    }

    private double ReLU(double value)
    {
        return System.Math.Max(0, value); // value < 0 ? 0 : value;
    }

    private double LeakyReLU(double value)
    {
        if (value > 0)
        {
            return value;
        }
        else
        {
            return value * 0.01;
        }
    }

    private double ReLUDeri(double value)
    {
        if (value > 0)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }

    private double LeakyReLUDeri(double value)
    {
        if (value > 0)
        {
            return 1;
        }
        else
        {
            return 0.01;
        }
    }
    
    public static float NextGaussian()
    {
        float v1, v2, s;
        do
        {
            v1 = 2.0f * Random.Range(-1.0f, 1.0f);
            v2 = 2.0f * Random.Range(-1.0f, 1.0f);
            s = v1 * v1 + v2 * v2;
        } while (s >= 1.0f || s == 0f);

        s = Mathf.Sqrt((-2.0f * Mathf.Log(s)) / s);

        return v1 * s;
    }
}

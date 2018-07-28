using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ANN {

    private double alpha = 0.001;
    private int numInputs;
    private bool visualize;
    List<Layer> layers = new List<Layer>();

	public ANN(int numberOfInputs, bool visualizeNetwork = false)
    {
        this.numInputs = numberOfInputs;
        this.visualize = visualizeNetwork;
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
        
        layers.Add(new Layer(numberOfNeurons, numberOfInputs, activationFunction, visualize));
    }


    public List<double> Predict(List<double> inputValues)
    {
        List<double> inputs;
        List<double> outputs = new List<double>();

        if (inputValues.Count != numInputs)
        {
            Debug.LogError("Number of Inputs must be " + numInputs);
            return outputs;
        }

        // step through every layer
        for (int i = 0; i < layers.Count; i++)
        {
            // on first layer we use the actual inputs
            if (i == 0)
            {
                inputs = new List<double>(inputValues);
            }
            // on every other layer we use the outputs we got so far
            else
            {
                inputs = new List<double>(outputs);
            }
            outputs.Clear();

            // step through every neuron in this layer
            for (int j = 0; j < layers[i].neurons.Count; j++)
            {
                // calculate neuron output and add it to output list for this layer
                outputs.Add(layers[i].neurons[j].CalculateOutput(inputs));
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

    public void TrainBatch(List<List<double>> inputValues, List<List<double>> desiredOutputValues)
    {
        /*
        List<List<double>> errorsList = new List<List<double>>();
        List<List<double>> outputsList = new List<List<double>>();

        for (int i = 0; i < inputValues.Count; i++)
        {
            List<double> errors = new List<double>();
            List<double> outputs = Predict(inputValues[i]);
            for (int j = 0; j < outputs.Count; j++)
            {
                // mean squared error
                //errors.Add(Mathf.Pow((float)(desiredOutputValues[i][j] - outputs[j]), 2));
                // mean error
                errors.Add(desiredOutputValues[i][j] - outputs[j]);
            }
            errorsList.Add(errors);
            outputsList.Add(outputs);
        }

        List<double> meanErrors = new List<double>();
        List<double> meanOutputs = new List<double>();
        for (int i = 0; i < errorsList[0].Count; i++)
        {
            meanErrors.Add(0);
            meanOutputs.Add(0);
        }

        for (int i = 0; i < errorsList.Count; i++)
        {
            for (int j = 0; j < errorsList[i].Count; j++)
            {
                meanErrors[j] += errorsList[i][j];
                meanOutputs[j] += outputsList[i][j];
            }
        }

        for (int i = 0; i < meanErrors.Count; i++)
        {
            meanOutputs[i] /= errorsList.Count;
            meanErrors[i] /= errorsList.Count;
            meanErrors[i] += meanOutputs[i];
        }

        UpdateWeights(meanOutputs, meanErrors);*/

        
        for (int i = 0; i < inputValues.Count; i++)
        {
            Train(inputValues[i], desiredOutputValues[i]);
        }
        
    }

    public void UpdateWeights(List<double> outputs, List<double> desiredOutputValues)
    {
        double error;

        // step through every layer BACKWARDS
        for (int i = layers.Count - 1; i >= 0; i--)
        {
            // step through every neuron in this layer
            for (int j = 0; j < layers[i].neurons.Count; j++)
            {
                // neuron is at the output layer
                if (i == (layers.Count - 1))
                {
                    // calculate the error for one neuron
                    error = desiredOutputValues[j] - outputs[j];
                    // calculate the error gradient
                    layers[i].neurons[j].errorGradient = outputs[j] * (1 - outputs[j]) * error;
                }
                // neuron is at any other layer
                else
                {
                    double errorGradSum = 0;
                    for (int p = 0; p < layers[i + 1].neurons.Count; p++)
                    {
                        // sum up the error gradients of -every neuron at the following layer- multiplied by -the weight of this neurons output-
                        errorGradSum += layers[i + 1].neurons[p].errorGradient * layers[i + 1].neurons[p].weights[j];
                    }
                    // caluclate the error gradient with regards to the error gradients of the connected neurons
                    layers[i].neurons[j].errorGradient = layers[i].neurons[j].output * (1 - layers[i].neurons[j].output) * errorGradSum;
                }

                // update all weights of this neuron regarding the calculated gradients
                for(int k = 0; k < layers[i].neurons[j].inputs.Count; k++)
                {
                    // neuron is at the output layer
                    if (i == (layers.Count - 1))
                    {
                        // calculate the error for one neuron
                        error = desiredOutputValues[j] - outputs[j];

                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * error;
                    }
                    // neuron is at any other layer
                    else
                    {
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * layers[i].neurons[j].errorGradient;
                    }
                }
                
                layers[i].neurons[j].bias += alpha * layers[i].neurons[j].errorGradient;
            }
        }
    }
}

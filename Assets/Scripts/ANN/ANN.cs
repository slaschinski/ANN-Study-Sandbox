using System.Collections;
using System.Collections.Generic;
using System.Xml.Linq;
using UnityEngine;

public class ANN {

    private double alpha;
    private double alphaDecay;
    private double lambda;
    private int numInputs;
    private DebugNeuronSpawner debugNeuronSpawner;
    private List<GameObject> debugInputNeurons;
    List<Layer> layers = new List<Layer>();

    public ANN(int numberOfInputs, double alpha, double alphaDecay = 1.0d, double lambda = 0.0d, GameObject debugNeuronSpawnerGameObject = null)
    {
        numInputs = numberOfInputs;
        this.alpha = alpha;
        this.alphaDecay = alphaDecay;
        this.lambda = lambda;

        // debug output visualization
        debugInputNeurons = new List<GameObject>();
        if (debugNeuronSpawnerGameObject != null) {
            debugNeuronSpawner = debugNeuronSpawnerGameObject.GetComponent<DebugNeuronSpawner>();
            for (int i = 0; i < numberOfInputs; i++) {
                debugInputNeurons.Add(debugNeuronSpawner.addNeuron(i, numberOfInputs, 0));
            }
        }
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
        
        layers.Add(new Layer(numberOfNeurons, numberOfInputs, activationFunction, debugNeuronSpawner));
    }


    public List<double> Predict(List<double> inputValues)
    {

        List<double> inputs;
        List<double> outputs = new List<double>();

        if (inputValues.Count != numInputs)
        {
            Debug.LogError("Number of Inputs must be " + numInputs);
            return outputs; // empty list
        }

        // step through every layer
        for (int i = 0; i < layers.Count; i++)
        {
            // on first layer we use the actual inputs
            if (i == 0)
            {
                inputs = new List<double>(inputValues);

                // debug output visualization
                if (debugInputNeurons.Count > 0)
                {
                    for (int j = 0; j < inputValues.Count; j++)
                    {
                        debugInputNeurons[j].GetComponent<DebugNeuron>().setOutput((float)inputValues[j]); 
                    }
                }
            }
            // on every other layer we use the outputs of the previous layer
            else
            {
                inputs = new List<double>(outputs);
            }
            outputs.Clear();

            // step through every neuron in this layer
            for (int j = 0; j < layers[i].neurons.Count; j++)
            {
                // set inputs of particular neuron
                layers[i].neurons[j].setInputs(inputs);
                // calculate neuron output and add it to output list for this layer
                outputs.Add(layers[i].neurons[j].CalculateOutput());
            }
        }
        
        return outputs;
    }

    public List<double> Train(List<double> inputValues, List<double> desiredOutputValues)
    {
        List<double> outputs = Predict(inputValues);

        UpdateWeights(outputs, desiredOutputValues);

        alpha *= alphaDecay;

        return outputs;
    }

    /*public void TrainBatch(List<List<double>> inputValues, List<List<double>> desiredOutputValues)
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

        /*
        for (int i = 0; i < inputValues.Count; i++)
        {
            Train(inputValues[i], desiredOutputValues[i]);
        }
        
    }*/

    public void UpdateWeights(List<double> outputs, List<double> desiredOutputValues)
    {

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
                    double error = desiredOutputValues[j] - outputs[j];
                    // calculate the error gradient
                    layers[i].neurons[j].CalculateErrorGradient(error);
                }
                // neuron is at any other layer
                else
                {
                    double errorGradSum = 0;
                    for (int p = 0; p < layers[i + 1].neurons.Count; p++)
                    {
                        // sum up the error gradients of -every neuron at the following layer- multiplied by -the weight for this neuron-
                        errorGradSum += layers[i + 1].neurons[p].getErrorGradient() * layers[i + 1].neurons[p].getWeights()[j];
                    }
                    // caluclate the error gradient with regards to the error gradients of the connected neurons
                    layers[i].neurons[j].CalculateErrorGradient(errorGradSum);
                }

                // update all weights of this neuron regarding the calculated gradients
                List<double> weights = new List<double>(layers[i].neurons[j].getWeights());
                for (int k = 0; k < layers[i].neurons[j].getInputs().Count; k++)
                {
                    /*// neuron is at the output layer
                    if (i == (layers.Count - 1))
                    {
                        // calculate the error for one neuron
                        double error = desiredOutputValues[j] - outputs[j];

                        weights[k] += alpha * layers[i].neurons[j].getInputs()[k] * error;
                    }
                    // neuron is at any other layer
                    else
                    {*/
                    weights[k] -= lambda * weights[k]; // L2 regularization
                    weights[k] += alpha * layers[i].neurons[j].getErrorGradient() * layers[i].neurons[j].getInputs()[k];
                    //}
                }
                layers[i].neurons[j].setWeights(weights);

                double bias = layers[i].neurons[j].getBias();
                bias -= lambda * bias; // L2 regularization
                bias += alpha * layers[i].neurons[j].getErrorGradient();
                layers[i].neurons[j].setBias(bias);
            }
        }
    }
}

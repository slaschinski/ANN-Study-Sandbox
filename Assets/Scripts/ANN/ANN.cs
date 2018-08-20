using System.Collections;
using System.Collections.Generic;
using System.Xml.Linq;
using UnityEngine;

/// <summary>
/// The neural network class
/// Combines neurons into layers and coordinates them to form a neural network
/// Handles back-propagation and is able to save the present state into an xml file
/// </summary>
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

        calculateGradients(outputs, desiredOutputValues);
        calculateDeltas(outputs, desiredOutputValues);

        UpdateWeights();

        alpha *= alphaDecay;
        lambda *= alphaDecay;

        return outputs;
    }

    public void TrainBatch(List<List<double>> inputValues, List<List<double>> desiredOutputValues)
    {
        bool sumDeltas = false;
        for (int i = 0; i < inputValues.Count; i++)
        {
            if (i > 0)
            {
                sumDeltas = true;
            }
            List<double> outputs = Predict(inputValues[i]);

            calculateGradients(outputs, desiredOutputValues[i]);
            calculateDeltas(outputs, desiredOutputValues[i], sumDeltas);
        }

        UpdateWeights();

        alpha *= alphaDecay;
        lambda *= alphaDecay;
    }

    public void UpdateWeights()
    {

        // step through every layer
        for (int i = 0; i < layers.Count; i++)
        {
            // step through every neuron in this layer
            for (int j = 0; j < layers[i].neurons.Count; j++)
            {
                layers[i].neurons[j].ApplyDeltas(lambda);
            }
        }
    }

    private void calculateGradients(List<double> outputs, List<double> desiredOutputValues)
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
                        errorGradSum += layers[i + 1].neurons[p].getErrorGradient() * layers[i + 1].neurons[p].getWeight(j);
                    }
                    // caluclate the error gradient with regards to the error gradients of the connected neurons
                    layers[i].neurons[j].CalculateErrorGradient(errorGradSum);
                }
            }
        }
    }

    public void calculateDeltas(List<double> outputs, List<double> desiredOutputValues, bool sumDeltas = false)
    {
        // step through every layer
        for (int i = 0; i < layers.Count; i++)
        {
            // step through every neuron in this layer
            for (int j = 0; j < layers[i].neurons.Count; j++)
            {
                // update all deltas of this neuron regarding the calculated gradients
                List<double> deltas = new List<double>();
                for (int k = 0; k < layers[i].neurons[j].getWeightCount(); k++)
                {
                    // neuron is at the output layer
                    if (i == (layers.Count - 1))
                    {
                        // calculate the error for one neuron
                        double error = desiredOutputValues[j] - outputs[j];
                        double delta = alpha * layers[i].neurons[j].getInput(k) * error;
                        //double delta = alpha * layers[i].neurons[j].getErrorGradient();
                        //double delta = alpha * layers[i].neurons[j].getErrorGradient() * layers[i].neurons[j].getInput(k);

                        deltas.Add(delta);
                    }
                    // neuron is at any other layer
                    else
                    {
                        double delta = alpha * layers[i].neurons[j].getErrorGradient() * layers[i].neurons[j].getInput(k);
                        deltas.Add(delta);
                    }
                }

                double biasDelta = alpha * layers[i].neurons[j].getErrorGradient();

                if (!sumDeltas)
                {
                    layers[i].neurons[j].setDeltas(deltas);
                    layers[i].neurons[j].setBiasDelta(biasDelta);
                } else
                {
                    layers[i].neurons[j].addToDeltas(deltas);
                    layers[i].neurons[j].addToBiasDelta(biasDelta);
                }



            }
        }
    }

    private XDocument brainToXml()
    {
        XElement brainXml = new XElement("Brain");
        brainXml.Add(new XElement("Inputs", numInputs));
        XDocument brainDoc = new XDocument(new XDeclaration("1.0", "UTF-16", null), brainXml);

        for (int i = 0; i < layers.Count; i++)
        {
            XElement layerXml = new XElement("Layer");
            layerXml.Add(new XElement("LayerID", i + 1));
            brainXml.Add(layerXml);

            for (int j = 0; j < layers[i].neurons.Count; j++)
            {
                XElement neuronXml = new XElement("Neuron");
                layerXml.Add(neuronXml);

                List<double> weights = layers[i].neurons[j].getWeights();
                for (int k = 0; k < weights.Count; k++)
                {
                    XElement weightXml = new XElement("Weight", weights[k]);
                    neuronXml.Add(weightXml);
                }
                XElement biasXml = new XElement("Bias", layers[i].neurons[j].getBias());
                neuronXml.Add(biasXml);
            }
        }

        return brainDoc;
    }

    private void XmlToBrain(XDocument brainDoc)
    {
        XElement brainXml = brainDoc.Element("Brain");
        IEnumerable<XElement> layersEnum = brainXml.Elements("Layer");
        int i = 0;
        foreach (XElement layerElement in layersEnum)
        {
            IEnumerable<XElement> neuronsEnum = layerElement.Elements("Neuron");
            int j = 0;
            foreach (XElement neuronElement in neuronsEnum)
            {
                IEnumerable<XElement> weightsEnum = neuronElement.Elements("Weight");
                List<double> newWeights = new List<double>();
                foreach (XElement weightElement in weightsEnum)
                {
                    newWeights.Add(double.Parse(weightElement.Value));
                }
                layers[i].neurons[j].setWeights(newWeights);
                layers[i].neurons[j].setBias(double.Parse(neuronElement.Element("Bias").Value));
                j++;
            }
            i++;
        }
    }

    public void saveBrain()
    {
        brainToXml().Save("D:\\savedBrain.xml");
    }

    public void loadBrain()
    {
        XDocument xdocument = XDocument.Load("D:\\loadBrain.xml");
        XmlToBrain(xdocument);
    }
}

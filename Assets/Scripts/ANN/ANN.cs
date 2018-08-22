using System.Collections;
using System.Collections.Generic;
using System.Xml.Linq;
using UnityEngine;

/// <summary>
/// The neural network class
/// Combines neurons into layers and coordinates them to form a neural network.
/// Handles back-propagation and is able to save the present state into an xml file.
/// </summary>
public class ANN {

    /// <summary>
    /// Learning rate
    /// </summary>
    private double alpha;
    /// <summary>
    /// Learning rate decay (alpha *= alphaDecay)
    /// Also decays lambda by the same rate.
    /// </summary>
    private double alphaDecay;
    /// <summary>
    /// Decay of weights (weight -= weight * lambda)
    /// </summary>
    private double lambda;
    /// <summary>
    /// Number of expected inputs
    /// </summary>
    private int numInputs;
    /// <summary>
    /// A special spawner to visualize neurons
    /// </summary>
    private DebugNeuronSpawner debugNeuronSpawner;
    /// <summary>
    /// Input neurons differ from every other neuron, so they get saved seperately
    /// </summary>
    private List<GameObject> debugInputNeurons;
    /// <summary>
    /// A list containing all layer objects, which contain the neurons
    /// </summary>
    List<Layer> layers = new List<Layer>();

    /// <summary>
    /// This constructor prepared the neural network by setting the hyperparameters, as well as the inputs
    /// and prepares the visualization of the neural net, if an inworld spawner for the neurons is given.
    /// Alpha indicates the learning rate, while it will be multiplied with alphaDecay after every learning
    /// cycle to gradually decay the learning rate. alphaDecay = 1.0 disables the decay.
    /// Lambda indicates a rate by which the neurons weights decay every learning cycle. This is a technique
    /// to combat overfitting and exploding gradients.
    /// </summary>
    /// <param name="numberOfInputs">An integer indicating the number of inputs to expect</param>
    /// <param name="alpha">A double representing the learning rate</param>
    /// <param name="alphaDecay">A double to reduce alpha (and lambda) over time by multiplication</param>
    /// <param name="lambda">A double to reduce weights by the amount of lambda multiplied with the actual weight</param>
    /// <param name="debugNeuronSpawnerGameObject">A gameobject variable containing a special spawner for neuron visualization if needed</param>
    public ANN(int numberOfInputs, double alpha, double alphaDecay = 1.0d, double lambda = 0.0d, GameObject debugNeuronSpawnerGameObject = null)
    {
        this.numInputs = numberOfInputs;
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

    /// <summary>
    /// Adds one layer to the neural network. It's an output layer as long as you don't add another layer.
    /// </summary>
    /// <param name="numberOfNeurons">An integer with the number of neurons this layer will contain</param>
    /// <param name="activationFunction">The name of the activation function to use. Defaults to ReLU.</param>
    public void AddLayer(int numberOfNeurons, string activationFunction = "ReLU")
    {
        int numberOfInputs;

        // on the first layer of calculating neurons we use the number of input neurons
        if (layers.Count == 0)
        {
            numberOfInputs = this.numInputs;
        }
        // on all the not-first layers we use the number of outputs of the previous layer
        else
        {
            numberOfInputs = layers[layers.Count - 1].neurons.Count;
        }
        
        layers.Add(new Layer(numberOfNeurons, numberOfInputs, activationFunction, debugNeuronSpawner));
    }

    /// <summary>
    /// Calculates the ANN outputs for the given inputs.
    /// </summary>
    /// <param name="inputValues">A list of double values used as inputs</param>
    /// <returns>A list a double values calculated by the ANN</returns>
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
            // on first calculating layer we use the actual inputs
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
            // on all the not-first layers we use the outputs of the previous layer as inputs
            else
            {
                // uses the outputs of the previous iteration as inputs
                inputs = new List<double>(outputs);
                // since this is at least the second iteration, we need to clean up the outputs before calculating them again
                outputs.Clear();
            }

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

    /// <summary>
    /// Trains the ANN by giving inputs and the desired output values for those inputs.
    /// </summary>
    /// <param name="inputValues">A list of double values used as inputs</param>
    /// <param name="desiredOutputValues">A list of double values which mark the desired outputs</param>
    /// <returns>A list of double values which are the calculated output before training</returns>
    public List<double> Train(List<double> inputValues, List<double> desiredOutputValues)
    {
        List<double> outputs = Predict(inputValues);

        calculateGradients(outputs, desiredOutputValues);
        calculateDeltas(outputs, desiredOutputValues);

        UpdateWeights();

        return outputs;
    }

    /// <summary>
    /// Trains the ANN by giving a batch of input values and corresponding desired output values.
    /// All calculated deltas will be summed up before the weights are updated.
    /// </summary>
    /// <param name="inputValues">A list of a list containing doubles which are used as inputs</param>
    /// <param name="desiredOutputValues">A list of a list containing doubles which contain the desired output values</param>
    public void TrainBatch(List<List<double>> inputValues, List<List<double>> desiredOutputValues)
    {
        bool sumDeltas = false;

        for (int i = 0; i < inputValues.Count; i++)
        {
            // after the first iteration the deltas are cleanly overwritten, so we can start to sum up now
            if (i > 0)
            {
                sumDeltas = true;
            }
            List<double> outputs = Predict(inputValues[i]);

            calculateGradients(outputs, desiredOutputValues[i]);
            calculateDeltas(outputs, desiredOutputValues[i], sumDeltas);
        }

        UpdateWeights();
    }

    /// <summary>
    /// Actually updates the weights and reduces the alpha and lambda values if decay is set.
    /// </summary>
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

        // decay alpha and lambda on the same rate, to prevent smaller alpha than lambda
        // otherweise weights would decay faster than they are relearned, at some point
        alpha *= alphaDecay;
        lambda *= alphaDecay;
    }

    /// <summary>
    /// Calculates the gradients for gradient descent and saves them in the neuron objects.
    /// The gradient is calculated based on the desired outputs in contrast to the actual outputs.
    /// </summary>
    /// <param name="outputs">A list of doubles containing the actual outputs of the neural network</param>
    /// <param name="desiredOutputValues">A list of doubles containing the desired outputs of the neural network</param>
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
                // neuron is at any not-output layer
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

    /// <summary>
    /// Calculates the deltas (differences) of current weights and corresponding gradients and saves them into
    /// the neuron objects.
    /// The boolean sumDeltas need to be set to false to get the actual deltas. This should be the case for online
    /// learning or the first dataset of a batch. For every other dataset of a batch sumDelta should be set to true.
    /// </summary>
    /// <param name="outputs">A list of doubles containing the actual outputs of the neural network</param>
    /// <param name="desiredOutputValues">A list of doubles containing the desired outputs of the neural network</param>
    /// <param name="sumDeltas">A boolean indicating whether deltas should be added to the sum or overwrite the sum</param>
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
                        // on the output layer the error is used directly to calculate the delta
                        double error = desiredOutputValues[j] - outputs[j];
                        double delta = alpha * layers[i].neurons[j].getInput(k) * error;

                        deltas.Add(delta);
                    }
                    // neuron is not at the output layer
                    else
                    {
                        // calculate delta of a neuron based on the gradient saved in the neuron object
                        double delta = alpha * layers[i].neurons[j].getErrorGradient() * layers[i].neurons[j].getInput(k);

                        deltas.Add(delta);
                    }
                }

                // delta for the bias is always calculated by the gradient saved in the neuron object
                double biasDelta = alpha * layers[i].neurons[j].getErrorGradient();

                // overwrite old deltas
                if (!sumDeltas)
                {
                    layers[i].neurons[j].setDeltas(deltas);
                    layers[i].neurons[j].setBiasDelta(biasDelta);
                }
                // sum up new deltas and saved deltas
                else
                {
                    layers[i].neurons[j].addToDeltas(deltas);
                    layers[i].neurons[j].addToBiasDelta(biasDelta);
                }



            }
        }
    }

    /// <summary>
    /// Convertes the ANN weights to an XML document which can be saved for later restore.
    /// </summary>
    /// <returns>An XDocument containing neuron weight values</returns>
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

    /// <summary>
    /// Restores the ANN weights from an XML document.
    /// </summary>
    /// <param name="brainDoc">An XDocument containing neuron weight values</param>
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

    /// <summary>
    /// Saves a saved brain to a file (Work in progress)
    /// </summary>
    public void saveBrain()
    {
        brainToXml().Save("D:\\savedBrain.xml");
    }

    /// <summary>
    /// Loads a saved brain from a file (Work in progress)
    /// </summary>
    public void loadBrain()
    {
        XDocument xdocument = XDocument.Load("D:\\loadBrain.xml");
        XmlToBrain(xdocument);
    }
}

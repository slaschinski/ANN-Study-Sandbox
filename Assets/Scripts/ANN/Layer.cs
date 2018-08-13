using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Layer {

    public List<Neuron> neurons = new List<Neuron>();

    public Layer(int numberOfNeurons, int numberOfInputsPerNeuron, string activationFunctionName, DebugNeuronSpawner debugNeuronSpawner)
    {
        if (debugNeuronSpawner != null)
        {
            debugNeuronSpawner.addLayer();
        }
        for (int i = 0; i < numberOfNeurons; i++)
        {
            GameObject debugNeuron = null;
            if (debugNeuronSpawner != null)
            {
                debugNeuron = debugNeuronSpawner.addNeuron(i, numberOfNeurons, numberOfInputsPerNeuron);
            }
            neurons.Add(new Neuron(numberOfInputsPerNeuron, numberOfNeurons, activationFunctionName, debugNeuron));
        }
    }
}

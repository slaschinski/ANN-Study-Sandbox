using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Layer {

    public List<Neuron> neurons = new List<Neuron>();

    public Layer(int numberOfNeurons, int numberOfInputsPerNeuron, string activationFunctionName, bool visualizeNetwork = false)
    {
        for (int i = 0; i < numberOfNeurons; i++)
        {
            neurons.Add(new Neuron(numberOfInputsPerNeuron, activationFunctionName));
            if (visualizeNetwork == true)
            {
                Debug.Log("Erstelle " + (i + 1) + " von " + numberOfNeurons);
            }
        }
    }
}

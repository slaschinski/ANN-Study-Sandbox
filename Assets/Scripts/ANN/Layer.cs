using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Layer {

    public List<Neuron> neurons = new List<Neuron>();

    public Layer(int numberOfNeurons, int numberOfInputsPerNeuron)
    {
        for (int i = 0; i < numberOfNeurons; i++)
        {
            neurons.Add(new Neuron(numberOfInputsPerNeuron));
        }
    }
}

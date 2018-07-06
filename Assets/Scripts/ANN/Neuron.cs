using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Neuron {
    public double bias;
    public double output;
    public double errorGradient;
    public List<double> inputs = new List<double>();
    public List<double> weights = new List<double>();

    // Initialize bias and weights with random values
    public Neuron(int numberOfInputs)
    {
        for (int i = 0; i < numberOfInputs; i++)
        {
            weights.Add(UnityEngine.Random.Range(-1.0f, 1.0f));
        }
        bias = UnityEngine.Random.Range(-1.0f, 1.0f);
    }
}

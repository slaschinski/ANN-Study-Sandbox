using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DebugNeuron : MonoBehaviour {

    GameObject biasObject;
    List<GameObject> weightObjects;

    // Use this for initialization
    void Start ()
    {
    }
	
	// Update is called once per frame
	void Update () {
		
	}

    public void setOutput(float output)
    {
        if (output > 1)
        {
            output = 1;
        } else if (output < -1)
        {
            output = -1;
        }
        if (output >= 0)
        {
            GetComponent<Renderer>().material.color = new Color(0, output, 0);
        }
        else
        {
            GetComponent<Renderer>().material.color = new Color(output * -1, 0, 0);
        }
    }

    public void setWeights(List<float> weights)
    {
        for (int i = 0; i < weights.Count; i++)
        {
            setWeightWidth(i, weights[i]);
        }
    }

    public void setWeights(List<double> weights)
    {
        for (int i = 0; i < weights.Count; i++)
        {
            setWeightWidth(i, (float)weights[i]);
        }

    }

    public void setWeightWidth(int weightNum, float weight)
    {
        float width;
        if (weight >= 0)
        {
            width = weight;
        }
        else
        {
            width = weight * -1;
        }
        width /= 100.0f;
        Transform weightTransform = weightObjects[weightNum].GetComponent<Transform>();
        weightTransform.localScale = new Vector3(width, weightTransform.localScale.y, weightTransform.localScale.z);
    }

    public void setWeightedInputs(List<float> weights, List<float> inputs)
    {
        for (int i = 0; i < weights.Count; i++)
        {
            setWeightColor(i, weights[i] * inputs[i]);
        }
    }

    public void setWeightedInputs(List<double> weights, List<double> inputs)
    {
        for (int i = 0; i < weights.Count; i++)
        {
            setWeightColor(i, (float)(weights[i] * inputs[i]));
        }

    }

    public void setWeightColor(int weightNum, float weight)
    {
        Renderer weightRenderer = weightObjects[weightNum].GetComponent<Renderer>();
        if (weight >= 0)
        {
            weightRenderer.material.color = new Color(0, weight, 0);
        }
        else
        {
            weightRenderer.material.color = new Color(weight * -1, 0, 0);
        }
    }

    public void setWeightObjects(List<GameObject> weightObjects)
    {
        this.weightObjects = weightObjects;
    }

    public void setBiasObject(GameObject biasObject)
    {
        this.biasObject = biasObject;
    }

    public void setBiasColor(double bias)
    {
        if (bias >= 0)
        {
            biasObject.GetComponent<Renderer>().material.color = new Color(0, (float)bias, 0);
        }
        else
        {
            biasObject.GetComponent<Renderer>().material.color = new Color((float)bias * -1, 0, 0);
        }
    }
}

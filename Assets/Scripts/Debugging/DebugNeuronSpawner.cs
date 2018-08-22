using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DebugNeuronSpawner : MonoBehaviour {

    int layerCounter = 0;
    float sizeFactor = 1.5f;

    Vector3 worldOffset;

    void Start()
    {
        worldOffset = GetComponent<Transform>().position;
    }

    public void addLayer()
    {
        layerCounter++;
    }

    public GameObject addNeuron(int neuronNum, int numberOfNeurons, int numberOfInputsPerNeuron)
    {
        float offset = (numberOfNeurons * sizeFactor) / 2.0f;

        GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        cube.name = layerCounter + " - " + (neuronNum + 1);
        //cube.GetComponent<Transform>().parent = this.GetComponent<Transform>();
        Vector3 cubePosition = new Vector3((layerCounter * sizeFactor * 15) + 20, 0, (neuronNum * sizeFactor * -1) + offset);
        cube.GetComponent<Transform>().position = cubePosition + worldOffset;
        cube.GetComponent<Transform>().localScale = new Vector3(1f, 0.1f, 1f);
        cube.AddComponent<DebugNeuron>();
        cube.layer = 9;
        cube.transform.parent = transform;

        if (layerCounter > 0)
        {
            GameObject bias = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            bias.GetComponent<Transform>().localScale = new Vector3(0.75f, 0.1f, 0.75f);
            bias.GetComponent<Transform>().position = cubePosition + (Vector3.up * 0.1f) + worldOffset;
            bias.name = layerCounter + " - " + (neuronNum + 1) + " - bias";
            bias.layer = 9;
            bias.transform.parent = cube.transform;
            cube.GetComponent<DebugNeuron>().setBiasObject(bias);
        }

        List <GameObject> weights = new List<GameObject>();
        for (int i = 0; i < numberOfInputsPerNeuron; i++)
        {
            float inputOffset = (numberOfInputsPerNeuron * sizeFactor) / 2.0f;
            Vector3 targetPosition = new Vector3(((layerCounter - 1) * sizeFactor * 15) + 20, 0, (i * sizeFactor * -1) + inputOffset);
            Vector3 centerPosition = (cubePosition + targetPosition) * 0.5f;
            Vector3 targetDirection = targetPosition - centerPosition;
            GameObject weight = GameObject.CreatePrimitive(PrimitiveType.Plane);
            Transform weightTransform = weight.GetComponent<Transform>();
            float distance = Vector3.Distance(cubePosition, targetPosition);
            weightTransform.localScale = new Vector3(0.1f, 1f, distance * 0.1f);
            weightTransform.position = centerPosition + worldOffset;
            weightTransform.rotation = Quaternion.LookRotation(targetDirection);
            weightTransform.name = layerCounter + " - " + (neuronNum + 1) + " - " + (i + 1);
            weight.layer = 9;
            weight.transform.parent = cube.transform;
            weights.Add(weight);
        }
        cube.GetComponent<DebugNeuron>().setWeightObjects(weights);

        return cube;
    }
}

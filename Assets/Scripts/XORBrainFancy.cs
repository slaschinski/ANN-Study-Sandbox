﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class XORBrainFancy : MonoBehaviour {

    public GameObject neuronSpawner;
    ANN ann;
    double sumSquareError = 0;
    float delta = 0;
    int i = 0;

	// Use this for initialization
	void Start () {
        ann = new ANN(2, 0.2f, 0.999f, 0.0003f, neuronSpawner);
        ann.AddLayer(2, "LeakyReLU");
        ann.AddLayer(1, "Sigmoid");
    }

    List<double> Train(double i1, double i2, double o)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();
        inputs.Add(i1);
        inputs.Add(i2);
        outputs.Add(o);
        return (ann.Train(inputs, outputs));
    }
	
	// Update is called once per frame
	void Update ()
    {
        delta += Time.deltaTime * 10;
        List<double> result;

        if ((int)Mathf.Floor(delta) > i)
        {
            int run = i % 4;
            switch (run) {
                case 0:
                    Debug.Log("1 1 -> " + Train(1, 1, 0)[0]);
                    break;
                case 1:
                    Debug.Log("1 0 -> " + Train(1, 0, 1)[0]);
                    break;
                case 2:
                    Debug.Log("0 1 -> " + Train(0, 1, 1)[0]);
                    break;
                case 3:
                    Debug.Log("0 0 -> " + Train(0, 0, 0)[0]);
                    break;
            }

            if (i > 40000)
            {
                sumSquareError = 0;
                result = Train(1, 1, 0);
                sumSquareError += Mathf.Pow((float)result[0] - 0, 2);
                Debug.Log(" 1 1 " + result[0]);
                result = Train(1, 0, 1);
                sumSquareError += Mathf.Pow((float)result[0] - 1, 2);
                Debug.Log(" 1 0 " + result[0]);
                result = Train(0, 1, 1);
                sumSquareError += Mathf.Pow((float)result[0] - 1, 2);
                Debug.Log(" 0 1 " + result[0]);
                result = Train(0, 0, 0);
                sumSquareError += Mathf.Pow((float)result[0] - 0, 2);
                Debug.Log(" 0 0 " + result[0]);
                Debug.Log("SSE: " + sumSquareError);
                GetComponent<XORBrainFancy>().enabled = false;
            }

            i++;
        }
    }
}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Indicator : MonoBehaviour {

    public int qId;
    public GameObject brainContainer;
    private BalanceBrain brain;
    private Renderer rend;

    // Use this for initialization
    void Start () {
        brain = brainContainer.GetComponent<BalanceBrain>();
        rend = GetComponent<Renderer>();
    }
	
	// Update is called once per frame
	void Update () {
        if (brain.maxQIndex == qId)
        {
            rend.material.color = Color.yellow;
        } else
        {
            rend.material.color = Color.white;
        }
	}
}

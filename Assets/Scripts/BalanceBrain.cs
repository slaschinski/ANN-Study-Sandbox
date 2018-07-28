using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class Replay
{
    public List<double> states;
    public double reward;
    public int actionX;
    public int actionZ;

    public Replay(
        double xr,
        double zr,
        double ballx,
        double ballz,
        double ballvx,
        double ballvz,
        int iX,
        int iZ,
        double r
        )
    {
        states = new List<double>();
        states.Add(xr);
        states.Add(zr);
        states.Add(ballx);
        states.Add(ballz);
        states.Add(ballvx);
        states.Add(ballvz);
        actionX = iX;
        actionZ = iZ;
        reward = r;
    }
}

public class BalanceBrain : MonoBehaviour {

    public GameObject ball;
    public Transform neuronPrefab;

    ANN ann;

    float reward = 0.0f;
    List<Replay> replayMemory = new List<Replay>();
    int mCapacity = 10000;

    float discount = 0.99f;
    float exploreRate = 100.0f;
    float minExploreRate = 0.01f;
    float exploreDecay = 0.9995f;

    Vector3 ballStartPos;
    int failCount = 0;
    float tiltSpeed = 0.5f;


    float timer = 0;
    float maxBalanceTime = 0;
    float totalReward = 0;


    List<double> qs = new List<double>();
    List<double> qsX = new List<double>();
    List<double> qsZ = new List<double>();
    public int maxQIndexX = 0;
    public int maxQIndexZ = 0;

    // Use this for initialization
    void Start () {
        ann = new ANN(6, true);
        ann.AddLayer(32, "ReLU");
        ann.AddLayer(16, "ReLU");
        ann.AddLayer(4, "Sigmoid");

        ballStartPos = ball.transform.position;
        Time.timeScale = 5.0f;
	}

    GUIStyle guiStyle = new GUIStyle();
    void OnGUI()
    {
        guiStyle.fontSize = 25;
        guiStyle.normal.textColor = Color.white;
        GUI.BeginGroup(new Rect(10, 10, 600, 150));
        GUI.Box(new Rect(0, 0, 140, 140), "Stats", guiStyle);
        GUI.Label(new Rect(10, 25, 500, 30), "Fails: " + failCount, guiStyle);
        GUI.Label(new Rect(10, 50, 500, 30), "Explore Rate: " + exploreRate, guiStyle);
        GUI.Label(new Rect(10, 75, 500, 30), "Last Best Balance: " + maxBalanceTime, guiStyle);
        GUI.Label(new Rect(10, 100, 500, 30), "This Balance: " + timer, guiStyle);
        GUI.Label(new Rect(10, 125, 500, 30), "Reward: " + totalReward, guiStyle);
        GUI.EndGroup();

        GUI.BeginGroup(new Rect(Screen.width-310, 10, 300, 450));
        GUI.Box(new Rect(0, 0, 140, 140), "Input", guiStyle);
        GUI.Label(new Rect(10, 25, 200, 30), "Plane X: " + this.transform.rotation.x, guiStyle);
        GUI.Label(new Rect(10, 50, 200, 30), "Plane Z: " + this.transform.rotation.z, guiStyle);
        GUI.Label(new Rect(10, 75, 200, 30), "Ball Pos X: " + ball.transform.position.x, guiStyle);
        GUI.Label(new Rect(10, 100, 200, 30), "Ball Pos Z: " + ball.transform.position.z, guiStyle);
        GUI.Label(new Rect(10, 125, 200, 30), "Ball Vel X: " + ball.GetComponent<Rigidbody>().velocity.x, guiStyle);
        GUI.Label(new Rect(10, 150, 200, 30), "Ball Vel Z: " + ball.GetComponent<Rigidbody>().velocity.z, guiStyle);
        GUI.Box(new Rect(0, 225, 140, 140), "Output", guiStyle);

        if (maxQIndexX == 0)
        {
            guiStyle.normal.textColor = Color.yellow;
        }
        GUI.Label(new Rect(10, 250, 200, 30), "X+: " + BarDiagram(qsX[0]), guiStyle);
        guiStyle.normal.textColor = Color.white;
        if (maxQIndexX == 1)
        {
            guiStyle.normal.textColor = Color.yellow;
        }
        GUI.Label(new Rect(10, 275, 200, 30), "X-:  " + BarDiagram(qsX[1]), guiStyle);
        guiStyle.normal.textColor = Color.white;
        if (maxQIndexZ == 2)
        {
            guiStyle.normal.textColor = Color.yellow;
        }
        GUI.Label(new Rect(10, 300, 200, 30), "Z+: " + BarDiagram(qsZ[0]), guiStyle);
        guiStyle.normal.textColor = Color.white;
        if (maxQIndexZ == 3)
        {
            guiStyle.normal.textColor = Color.yellow;
        }
        GUI.Label(new Rect(10, 325, 200, 30), "Z-:  " + BarDiagram(qsZ[1]), guiStyle);
        guiStyle.normal.textColor = Color.white;

        GUI.EndGroup();
    }

    // Update is called once per frame
    void Update () {
        if (Input.GetKeyDown("space"))
        {
            ResetState();
        }
    }

    void FixedUpdate()
    {
        timer += Time.deltaTime;
        List<double> states = new List<double>();

        states.Add(this.transform.rotation.x);
        states.Add(this.transform.rotation.z);
        states.Add(ball.transform.position.x);
        states.Add(ball.transform.position.z);
        states.Add(ball.GetComponent<Rigidbody>().velocity.x);
        states.Add(ball.GetComponent<Rigidbody>().velocity.z);

        qsX.Clear();
        qsZ.Clear();
        qs = ann.Predict(states);
        qsX.Add(qs[0]);
        qsX.Add(qs[1]);
        qsZ.Add(qs[2]);
        qsZ.Add(qs[3]);
        qsX = SoftMax(qsX);
        qsZ = SoftMax(qsZ);
        double maxQX = qsX.Max();
        double maxQZ = qsZ.Max();
        maxQIndexX = qsX.ToList().IndexOf(maxQX);
        maxQIndexZ = qsZ.ToList().IndexOf(maxQZ) + 2;

        if (Random.Range(1, 100) < exploreRate)
        {
            maxQIndexX = Random.Range(0, 2);
            maxQIndexZ = Random.Range(2, 4);
        }

        if (maxQIndexX == 0)
        {
            transform.Rotate(Vector3.right, tiltSpeed); //* (float)qs[maxQIndexX]);
        }
        else //if (maxQIndexX == 1)
        {
            transform.Rotate(Vector3.right, -tiltSpeed); //* (float)qs[maxQIndexX]);
        }

        if (maxQIndexZ == 2)
        {
            transform.Rotate(Vector3.forward, tiltSpeed); //* (float)qs[maxQIndexZ]);
        }
        else //if (maxQIndexZ == 3)
        {
            transform.Rotate(Vector3.forward, -tiltSpeed); //* (float)qs[maxQIndexZ]);
        }

        if (ball.GetComponent<BallState>().dropped)
        {
            reward = -1.0f;// - ball.GetComponent<Rigidbody>().velocity.magnitude;
        }
        else
        {
            reward = 0.01f;
        }
        totalReward += reward;

        Replay lastMemory = new Replay(this.transform.rotation.x,
                                       this.transform.rotation.z,
                                       ball.transform.position.x,
                                       ball.transform.position.z,
                                       ball.GetComponent<Rigidbody>().angularVelocity.x,
                                       ball.GetComponent<Rigidbody>().angularVelocity.z,
                                       maxQIndexX,
                                       maxQIndexZ,
                                       reward);

        if (replayMemory.Count > mCapacity)
        {
            replayMemory.RemoveAt(0);
        }

        replayMemory.Add(lastMemory);

        if (ball.GetComponent<BallState>().dropped || replayMemory.Count > 1000)
        {
            if (exploreRate > minExploreRate)
            {
                exploreRate *= exploreDecay;
            }

            /*int batchSize = 32;
            if (replayMemory.Count < batchSize)
            {
                batchSize = replayMemory.Count;
            }

            List<int> samples = new List<int>();
            while (samples.Count < batchSize)
            {
                int rand = Random.Range(0, replayMemory.Count);
                if (!samples.Contains(rand))
                {
                    samples.Add(rand);
                }
            }

            List<List<double>> batchInputs = new List<List<double>>();
            List<List<double>> batchOutputs = new List<List<double>>();
            */

            //for (int samp = 0; samp < samples.Count; samp++)
            for (int i = replayMemory.Count - 1; i >= 0; i--)
            {
                //int i = samples[samp];

                double feedbackX;
                double feedbackZ;

                if (i == replayMemory.Count - 1 || replayMemory[i].reward == -1)
                {
                    feedbackX = replayMemory[i].reward;
                    feedbackZ = replayMemory[i].reward;
                }
                else
                {
                    List<double> toutputsNext = new List<double>();
                    toutputsNext = ann.Predict(replayMemory[i + 1].states);
                    //double maxQNext = toutputsNext.Max();
                    List<double> toutputsNextX = new List<double>();
                    List<double> toutputsNextZ = new List<double>();
                    toutputsNextX.Add(toutputsNext[0]);
                    toutputsNextX.Add(toutputsNext[1]);
                    toutputsNextZ.Add(toutputsNext[2]);
                    toutputsNextZ.Add(toutputsNext[3]);
                    toutputsNextX = SoftMax(toutputsNextX);
                    toutputsNextZ = SoftMax(toutputsNextZ);
                    double maxQNextX = toutputsNextX.Max();
                    double maxQNextZ = toutputsNextZ.Max();
                    feedbackX = (replayMemory[i].reward + discount * maxQNextX);
                    feedbackZ = (replayMemory[i].reward + discount * maxQNextZ);
                }

                List<double> toutputsNow = new List<double>();
                toutputsNow = ann.Predict(replayMemory[i].states);
                List<double> toutputsNowOld = new List<double>(toutputsNow);

                List<double> toutputsNowX = new List<double>();
                List<double> toutputsNowZ = new List<double>();
                toutputsNowX.Add(toutputsNow[0]);
                toutputsNowX.Add(toutputsNow[1]);
                toutputsNowZ.Add(toutputsNow[2]);
                toutputsNowZ.Add(toutputsNow[3]);
                toutputsNowX = SoftMax(toutputsNowX);
                toutputsNowZ = SoftMax(toutputsNowZ);


                int actionNowX = replayMemory[i].actionX;
                int actionNowZ = replayMemory[i].actionZ;

                // thisQ = thisQ + learnRate * [thisReward + discount * nextQMax - thisQ];
                //toutputsOld[action] = 0.5f * Mathf.Pow((float)(feedback - maxQOld), 2.0f);
                toutputsNow[actionNowX] = toutputsNow[actionNowX] * (feedbackX - toutputsNowX[actionNowX]);
                toutputsNow[actionNowZ] = toutputsNow[actionNowZ] * (feedbackZ - toutputsNowZ[actionNowZ - 2]);
                ann.UpdateWeights(toutputsNowOld, toutputsNow);
                //batchInputs.Add(replayMemory[i].states);
                //batchOutputs.Add(toutputsNow);
            }
            //ann.TrainBatch(batchInputs, batchOutputs);


            if (ball.GetComponent<BallState>().dropped)
            {
                if (timer > maxBalanceTime)
                {
                    maxBalanceTime = timer;
                }

                timer = 0;
                totalReward = 0;

                ResetState();
                failCount++;
            }
            replayMemory.Clear();
        }
    }

    void ResetState()
    {
        transform.rotation = Quaternion.identity;
        transform.Rotate(Vector3.forward, Random.Range(-1.0f, 1.0f));
        transform.Rotate(Vector3.right, Random.Range(-1.0f, 1.0f));
        ball.GetComponent<BallState>().dropped = false;
        ball.transform.position = ballStartPos;
        ball.transform.position += new Vector3(Random.Range(-1.0f, 1.0f), 0, Random.Range(-1.0f, 1.0f));
        ball.GetComponent<Rigidbody>().velocity = new Vector3(0, 0, 0);  //new Vector3(Random.Range(-1.0f, 1.0f), 0, Random.Range(-1.0f, 1.0f));
        ball.GetComponent<Rigidbody>().angularVelocity = new Vector3(0, 0, 0);
    }

    List<double> SoftMax(List<double> values)
    {
        double max = values.Max();

        float scale = 0.0f;
        for (int i = 0; i < values.Count; i++)
        {
            scale += Mathf.Exp((float)(values[i] - max));
        }

        List<double> result = new List<double>();
        for (int i = 0; i < values.Count; i++)
        {
            result.Add(Mathf.Exp((float)(values[i] - max)) / scale);
        }

        return result;
    }

    string BarDiagram(double value)
    {
        int bars = (int)Mathf.Ceil((float)value * 50.0f);
        string returnValue = "";
        for (int i = 0; i < bars; i++)
        {
            returnValue += "|";
        }
        return returnValue;
    }


}

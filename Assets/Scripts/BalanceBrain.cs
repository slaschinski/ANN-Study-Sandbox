using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class Replay
{
    public List<double> states;
    public double reward;

    public Replay(
        double xr,
        double zr,
        double ballx,
        double bally,
        double ballz,
        double ballvx,
        double ballvy,
        double ballvz,
        double r
        )
    {
        states = new List<double>();
        states.Add(xr);
        states.Add(zr);
        states.Add(ballx);
        states.Add(bally);
        states.Add(ballz);
        states.Add(ballvx);
        states.Add(ballvy);
        states.Add(ballvz);
        reward = r;
    }
}

public class BalanceBrain : MonoBehaviour {

    public GameObject ball;

    ANN ann;

    float reward = 0.0f;
    List<Replay> replayMemory = new List<Replay>();
    int mCapacity = 10000;

    float discount = 0.9f;
    float exploreRate = 100.0f;
    float maxExploreRate = 100.0f;
    float minExploreRate = 0.01f;
    float exploreDecay = 0.001f;

    Vector3 ballStartPos;
    int failCount = 0;
    float tiltSpeed = 0.5f;


    float timer = 0;
    float maxBalanceTime = 0;
    float totalReward = 0;


    List<double> qs = new List<double>();

    // Use this for initialization
    void Start () {
        ann = new ANN(8);
        ann.AddLayer(16, "ReLU");
        ann.AddLayer(8, "ReLU");
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

        GUI.BeginGroup(new Rect(Screen.width-310, 10, 300, 350));
        GUI.Box(new Rect(0, 0, 140, 140), "Input", guiStyle);
        GUI.Label(new Rect(10, 25, 200, 30), "Plane X: " + this.transform.rotation.x, guiStyle);
        GUI.Label(new Rect(10, 50, 200, 30), "Plane Z: " + this.transform.rotation.z, guiStyle);
        GUI.Label(new Rect(10, 75, 200, 30), "Ball Pos X: " + ball.transform.position.x, guiStyle);
        GUI.Label(new Rect(10, 100, 200, 30), "Ball Pos Y: " + ball.transform.position.y, guiStyle);
        GUI.Label(new Rect(10, 125, 200, 30), "Ball Pos Z: " + ball.transform.position.z, guiStyle);
        GUI.Label(new Rect(10, 150, 200, 30), "Ball Vel X: " + ball.GetComponent<Rigidbody>().velocity.x, guiStyle);
        GUI.Label(new Rect(10, 175, 200, 30), "Ball Vel Y: " + ball.GetComponent<Rigidbody>().velocity.y, guiStyle);
        GUI.Label(new Rect(10, 200, 200, 30), "Ball Vel Z: " + ball.GetComponent<Rigidbody>().velocity.z, guiStyle);
        GUI.Box(new Rect(0, 225, 140, 140), "Output", guiStyle);
        GUI.Label(new Rect(10, 250, 200, 30), "0: " + BarDiagram(qs[0]), guiStyle);
        GUI.Label(new Rect(10, 275, 200, 30), "1: " + BarDiagram(qs[1]), guiStyle);
        GUI.Label(new Rect(10, 300, 200, 30), "2: " + BarDiagram(qs[2]), guiStyle);
        GUI.Label(new Rect(10, 325, 200, 30), "3: " + BarDiagram(qs[3]), guiStyle);
        GUI.EndGroup();
    }

    // Update is called once per frame
    void Update () {
        if (Input.GetKeyDown("space"))
        {
            ResetBall();
        }
    }

    void FixedUpdate()
    {
        timer += Time.deltaTime;
        List<double> states = new List<double>();

        states.Add(this.transform.rotation.x);
        states.Add(this.transform.rotation.z);
        states.Add(ball.transform.position.x);
        states.Add(ball.transform.position.y);
        states.Add(ball.transform.position.z);
        states.Add(ball.GetComponent<Rigidbody>().velocity.x);
        states.Add(ball.GetComponent<Rigidbody>().velocity.y);
        states.Add(ball.GetComponent<Rigidbody>().velocity.z);

        qs = SoftMax(ann.Predict(states));
        double maxQ = qs.Max();
        int maxQIndex = qs.ToList().IndexOf(maxQ);
        exploreRate = Mathf.Clamp(exploreRate - exploreDecay, minExploreRate, maxExploreRate);

        if (Random.Range(1, 100) < exploreRate)
        {
            maxQIndex = Random.Range(0, 4);
        }

        if (maxQIndex == 0)
        {
            transform.Rotate(Vector3.right, tiltSpeed * (float)qs[maxQIndex]);
        }
        else if (maxQIndex == 1)
        {
            transform.Rotate(Vector3.right, -tiltSpeed * (float)qs[maxQIndex]);
        }
        else if (maxQIndex == 2)
        {
            transform.Rotate(Vector3.forward, tiltSpeed * (float)qs[maxQIndex]);
        }
        else //if (maxQIndex == 3)
        {
            transform.Rotate(Vector3.forward, -tiltSpeed * (float)qs[maxQIndex]);
        }

        if (ball.GetComponent<BallState>().dropped)
        {
            reward = -1.0f;
        }
        else
        {
            reward = 0.1f;
        }
        totalReward += reward;

        Replay lastMemory = new Replay(this.transform.rotation.x,
                                       this.transform.rotation.z,
                                       ball.transform.position.x,
                                       ball.transform.position.y,
                                       ball.transform.position.z,
                                       ball.GetComponent<Rigidbody>().angularVelocity.x,
                                       ball.GetComponent<Rigidbody>().angularVelocity.y,
                                       ball.GetComponent<Rigidbody>().angularVelocity.z,
                                       reward);

        if (replayMemory.Count > mCapacity)
        {
            replayMemory.RemoveAt(0);
        }

        replayMemory.Add(lastMemory);

        if (ball.GetComponent<BallState>().dropped || replayMemory.Count > 1000)
        {
            for (int i = replayMemory.Count - 1; i >= 0; i--)
            {
                List<double> toutputsOld = new List<double>();
                List<double> toutputsNew = new List<double>();

                toutputsOld = SoftMax(ann.Predict(replayMemory[i].states));

                double maxQOld = toutputsOld.Max();
                int action = toutputsOld.ToList().IndexOf(maxQOld);

                double feedback;
                if (i == replayMemory.Count - 1 || replayMemory[i].reward == -1)
                {
                    feedback = replayMemory[i].reward;
                }
                else
                {
                    toutputsNew = SoftMax(ann.Predict(replayMemory[i + 1].states));
                    maxQ = toutputsNew.Max();
                    feedback = (replayMemory[i].reward + discount * maxQ);
                }

                toutputsOld[action] = feedback;
                ann.Train(replayMemory[i].states, toutputsOld);
            }

            if (timer > maxBalanceTime)
            {
                maxBalanceTime = timer;
            }

            timer = 0;
            totalReward = 0;

            ball.GetComponent<BallState>().dropped = false;
            this.transform.rotation = Quaternion.identity;
            ResetBall();
            replayMemory.Clear();
            failCount++;
        }
    }

    void ResetBall()
    {
        ball.transform.position = ballStartPos;
        ball.transform.position += new Vector3(Random.Range(-2.0f, 2.0f), 0, Random.Range(-2.0f, 2.0f));
        ball.GetComponent<Rigidbody>().velocity = new Vector3(0, 0, 0);
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
        int bars = (int)Mathf.Ceil((float)value * 20.0f);
        string returnValue = "";
        for (int i = 0; i < bars; i++)
        {
            returnValue += "|";
        }
        return returnValue;
    }
}

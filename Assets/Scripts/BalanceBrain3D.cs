using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class Replay3D
{
    public List<double> states;
    public double reward;
    public int action1;
    public int action2;

    public Replay3D(
        double xr,
        double yr,
        double zr,
        double ballx,
        double ballz,
        double ballavx,
        double ballavz,
        int a1,
        int a2,
        double r
        )
    {
        states = new List<double>();
        states.Add(xr);
        states.Add(yr);
        states.Add(zr);
        states.Add(ballx);
        states.Add(ballz);
        states.Add(ballavx);
        states.Add(ballavz);
        action1 = a1;
        action2 = a2;
        reward = r;
    }
}

public class BalanceBrain3D : MonoBehaviour
{
    public GameObject ball;
    public GameObject neuronSpawner;

    ANN ann;

    List<Replay3D> replayMemory = new List<Replay3D>();
    //int mCapacity = 10000;

    float discount = 0.995f;
    float exploreRate = 100.0f;
    float minExploreRate = 0.01f;
    float exploreDecay = 0.01f;
    //int exploreDuration = 10;
    //int exploreDurationLeft;
    //int exploreDirection;
    float learningRate = 0.000005f;
    float learningRateDecay = 0.999f;
    float weightsDecay = 0.00003f;

    Vector3 ballStartPos;
    int failCount = 0;
    float tiltSpeed = 0.25f;


    float timer = 0;
    float maxBalanceTime = 0;


    List<double> qs1 = new List<double>();
    List<double> qs2 = new List<double>();
    int maxQIndex1 = 0;
    int maxQIndex2 = 0;

    private List<float> balanceTimes = new List<float>();
    private float averageBalanceTime;

    // Use this for initialization
    void Start()
    {
        ann = new ANN(7, learningRate, learningRateDecay, weightsDecay, neuronSpawner);
        ann.AddLayer(15, "LeakyReLU");
        ann.AddLayer(25, "LeakyReLU");
        ann.AddLayer(15, "LeakyReLU");
        ann.AddLayer(4, "Linear");
        //ann.loadBrain();

        /*
        List<double> target = new List<double>();
        target.Add(0.1f); target.Add(-0.1f);
        List<double> input = new List<double>();
        input.Add(0.0f); input.Add(0.5f); input.Add(0.0f);
        ann.Train(input, target);
        input.Clear();
        input.Add(0.0f); input.Add(0.0f); input.Add(0.5f);
        ann.Train(input, target);
        input.Clear();
        input.Add(0.0f); input.Add(0.5f); input.Add(0.5f);
        ann.Train(input, target);

        target.Clear();
        target.Add(-0.1f); target.Add(0.1f);
        input.Clear();
        input.Add(0.0f); input.Add(-0.5f); input.Add(0.0f);
        ann.Train(input, target);
        input.Clear();
        input.Add(0.0f); input.Add(0.0f); input.Add(-0.5f);
        ann.Train(input, target);
        input.Clear();
        input.Add(0.0f); input.Add(0.5f); input.Add(-0.5f);
        ann.Train(input, target);
        */

        ballStartPos = ball.transform.position;
        Time.timeScale = 5.0f;
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown("space"))
        {
            ResetState();
        }
    }

    void FixedUpdate()
    {
        timer += Time.deltaTime;
        List<double> states = new List<double>();
        states.Add(this.transform.rotation.x * 10);
        states.Add(this.transform.rotation.y * 10);
        states.Add(this.transform.rotation.z * 10);
        states.Add(ball.transform.position.x / 2.5f);
        states.Add(ball.transform.position.z / 2.5f);
        states.Add(ball.GetComponent<Rigidbody>().velocity.x / 4.0f);
        states.Add(ball.GetComponent<Rigidbody>().velocity.z / 4.0f);

        List<double> qs = ann.Predict(states);
        qs1 = new List<double>();
        qs2 = new List<double>();
        qs1.Add(qs[0]);
        qs1.Add(qs[1]);
        qs2.Add(qs[2]);
        qs2.Add(qs[3]);
        List<double> qs1SoftMax = SoftMax(qs1);
        List<double> qs2SoftMax = SoftMax(qs2);
        double maxQ1 = qs1.Max();
        double maxQ2 = qs2.Max();
        maxQIndex1 = qs1.ToList().IndexOf(maxQ1);
        maxQIndex2 = qs2.ToList().IndexOf(maxQ2);

        if (Random.Range(1, 100) < exploreRate)
        {
            maxQIndex1 = Random.Range(0, 2);
            maxQIndex2 = Random.Range(0, 2);
        }

        if (maxQIndex1 == 0 && this.transform.rotation.z >= -0.25f)
        {
            transform.Rotate(Vector3.forward, tiltSpeed);// * (float)qs1SoftMax[maxQIndex1]);
        }
        else if (maxQIndex1 == 1 && this.transform.rotation.z <= 0.25f)
        {
            transform.Rotate(Vector3.forward, -tiltSpeed);// * (float)qs1SoftMax[maxQIndex1]);
        }

        if (maxQIndex2 == 0 && this.transform.rotation.x <= 0.25f)
        {
            transform.Rotate(Vector3.right, tiltSpeed);// * (float)qs2SoftMax[maxQIndex2]);
        }
        else if (maxQIndex2 == 1 && this.transform.rotation.x <= 0.25f)
        {
            transform.Rotate(Vector3.right, -tiltSpeed);// * (float)qs2SoftMax[maxQIndex2]);
        }

        float reward = 0.0f;
        if (ball.GetComponent<BallState>().dropped)
        {
            reward = -1.0f; // * (Mathf.Sqrt(Mathf.Pow(ball.GetComponent<Rigidbody>().angularVelocity.z, 2.0f)) / 4.0f);
        }
        else
        {
            reward = 0.1f; // - Mathf.Sqrt(Mathf.Pow(ball.transform.position.x, 2.0f));
        }

        if (maxQIndex1 != -1 && maxQIndex2 != -1)
        {
            Replay3D lastMemory = new Replay3D(states[0],
                                               states[1],
                                               states[2],
                                               states[3],
                                               states[4],
                                               states[5],
                                               states[6],
                                               maxQIndex1,
                                               maxQIndex2,
                                               reward);

            /*if (replayMemory.Count > mCapacity)
            {
                replayMemory.RemoveAt(0);
            }*/

            replayMemory.Add(lastMemory);
        }

        if (ball.GetComponent<BallState>().dropped || replayMemory.Count > 5000)
        {
            if (exploreRate > minExploreRate)
            {
                exploreRate -= exploreDecay;
            }

            int batchSize = 32;
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

            for (int samp = 0; samp < samples.Count; samp++)
            //for (int i = replayMemory.Count - 1; i >= 0; i--)
            {
                int i = samples[samp];

                double feedback1;
                double feedback2;

                if (i == replayMemory.Count - 1 || replayMemory[i].reward == -1)
                {
                    feedback1 = replayMemory[i].reward;
                    feedback2 = replayMemory[i].reward;
                }
                else
                {
                    List<double> toutputsNext = ann.Predict(replayMemory[i + 1].states);
                    List<double> toutputs1Next = new List<double>();
                    List<double> toutputs2Next = new List<double>();
                    toutputs1Next.Add(toutputsNext[0]);
                    toutputs1Next.Add(toutputsNext[1]);
                    toutputs2Next.Add(toutputsNext[2]);
                    toutputs2Next.Add(toutputsNext[3]);
                    double maxQ1Next = toutputs1Next.Max();
                    double maxQ2Next = toutputs2Next.Max();
                    feedback1 = (replayMemory[i].reward + (discount * maxQ1Next));
                    feedback2 = (replayMemory[i].reward + (discount * maxQ2Next));
                }

                List<double> toutputsNow = ann.Predict(replayMemory[i].states);
                //List<double> toutputsNowOld = new List<double>(toutputsNow);

                int action1Now = replayMemory[i].action1;
                int action2Now = replayMemory[i].action2;

                // thisQ = thisQ + learnRate * [thisReward + discount * nextQMax - thisQ];
                //toutputsOld[action] = 0.5f * Mathf.Pow((float)(feedback - maxQOld), 2.0f);
                //toutputsNow[actionNow] += feedback - toutputsNow[actionNow];
                //toutputsNow[actionNow] += feedback;
                toutputsNow[action1Now] = feedback1;
                toutputsNow[action2Now + 2] = feedback2;

                //ann.UpdateWeights(toutputsNowOld, toutputsNow);
                batchInputs.Add(replayMemory[i].states);
                batchOutputs.Add(toutputsNow);
            }
            ann.TrainBatch(batchInputs, batchOutputs);

            if (ball.GetComponent<BallState>().dropped)
            {
                if (timer > maxBalanceTime)
                {
                    maxBalanceTime = timer;
                }

                balanceTimes.Add(timer);
                if (balanceTimes.Count > 100)
                {
                    balanceTimes.RemoveAt(0);
                }

                averageBalanceTime = 0;
                foreach (float balanceTime in balanceTimes)
                {
                    averageBalanceTime += balanceTime;
                }
                averageBalanceTime /= balanceTimes.Count;

                timer = 0;

                failCount++;
            }
            ResetState();
            replayMemory.Clear();
            ann.saveBrain();
        }
    }

    GUIStyle guiStyle = new GUIStyle();
    void OnGUI()
    {
        guiStyle.fontSize = 25;
        guiStyle.normal.textColor = Color.white;

        GUI.BeginGroup(new Rect(Screen.width - 350, 10, 350, 550));
        GUI.Box(new Rect(0, 0, 350, 100), "Stats", guiStyle);
        GUI.Label(new Rect(10, 25, 350, 25), "Fails: " + failCount, guiStyle);
        GUI.Label(new Rect(10, 50, 350, 25), "Explore Rate: " + exploreRate, guiStyle);
        GUI.Label(new Rect(10, 75, 350, 25), "Last Best Balance: " + maxBalanceTime, guiStyle);
        GUI.Label(new Rect(10, 100, 350, 25), "This Balance: " + timer, guiStyle);
        GUI.Label(new Rect(10, 125, 350, 25), "Average Balance: " + averageBalanceTime, guiStyle);
        GUI.Box(new Rect(0, 150, 350, 75), "Input", guiStyle);
        GUI.Label(new Rect(10, 175, 350, 25), "Plane X: " + this.transform.rotation.x * 10, guiStyle);
        GUI.Label(new Rect(10, 200, 350, 25), "Plane Y: " + this.transform.rotation.y * 10, guiStyle);
        GUI.Label(new Rect(10, 225, 350, 25), "Plane Z: " + this.transform.rotation.z * 10, guiStyle);
        GUI.Label(new Rect(10, 250, 350, 25), "Ball Pos X: " + ball.transform.position.x / 2.5f, guiStyle);
        GUI.Label(new Rect(10, 275, 350, 25), "Ball Pos Z: " + ball.transform.position.z / 2.5f, guiStyle);
        GUI.Label(new Rect(10, 300, 350, 25), "Ball AVel X: " + ball.GetComponent<Rigidbody>().velocity.x / 4.0f, guiStyle);
        GUI.Label(new Rect(10, 325, 350, 25), "Ball AVel Z: " + ball.GetComponent<Rigidbody>().velocity.z / 4.0f, guiStyle);
        GUI.Box(new Rect(0, 375, 350, 25), "Output", guiStyle);

        if (maxQIndex1 == 0)
        {
            guiStyle.normal.textColor = Color.yellow;
        }
        List<double> probability1 = SoftMax(qs1);
        List<double> probability2 = SoftMax(qs2);
        GUI.Label(new Rect(10, 400, 350, 25), '\u25C4' + " : " + BarDiagram(probability1[0]), guiStyle);
        guiStyle.normal.textColor = Color.white;
        if (maxQIndex1 == 1)
        {
            guiStyle.normal.textColor = Color.yellow;
        }
        GUI.Label(new Rect(10, 425, 350, 25), '\u25BA' + " : " + BarDiagram(probability1[1]), guiStyle);
        guiStyle.normal.textColor = Color.white;
        if (maxQIndex2 == 0)
        {
            guiStyle.normal.textColor = Color.yellow;
        }
        GUI.Label(new Rect(10, 450, 350, 25), '\u25B2' + " : " + BarDiagram(probability2[0]), guiStyle);
        guiStyle.normal.textColor = Color.white;
        if (maxQIndex2 == 1)
        {
            guiStyle.normal.textColor = Color.yellow;
        }
        GUI.Label(new Rect(10, 475, 350, 25), '\u25BC' + " : " + BarDiagram(probability2[1]), guiStyle);
        guiStyle.normal.textColor = Color.white;

        GUI.EndGroup();
    }

    void ResetState()
    {
        transform.rotation = Quaternion.identity;
        transform.Rotate(Vector3.forward, Random.Range(-5.0f, 5.0f));
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
        int bars = (int)Mathf.Ceil((float)value * 20.0f);
        string returnValue = "";
        for (int i = 0; i < bars; i++)
        {
            returnValue += "|";
        }
        return returnValue;
    }

}

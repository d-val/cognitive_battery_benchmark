using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GravityBiasRunner : MonoBehaviour
{

    public GameObject VTube;
    public GameObject STube;
    public GameObject STubeLong;

    public int numReceptacles;

    public GameObject sceneCamera;
    public FrameRecorder recorder;
    
    public List<GameObject> apples;
    public int movingSpeed;
    public float targetYPos;
    
    public int seed;
    
    public Dictionary<string, float> coordinateRanges = new Dictionary<string, float>(){
        {"xmin", -2.05f},
        {"xmax", -1.85f},

        {"ymin", 15.5f},
        {"ymax", 16.5f},

        {"zmin", 2f},
        {"zmax", 2.2f},
    };

    private int curAppleIdx = 0;
    private Vector3 destVector;

    // An experimental condition describing the tube type, z-coordinate, and rotation over y-axis.
    private struct Condition
    {
        public string tubeType;
        public float tubeZPos;
        public float tubeYRot;

        public Condition(string tubeType, float tubeZPos, float tubeYRot){
            this.tubeType = tubeType;
            this.tubeZPos = tubeZPos;
            this.tubeYRot = tubeYRot;
        }
    }

    // Maps each possible drop/final pair into a game condition.
    private Dictionary<(int drop, int final), Condition> exptStates = new Dictionary<(int, int), Condition>(){
        {(0, 0), new Condition("vertical", 2.1f, 0f)},
        {(0, 1), new Condition("s_shaped", -0.57f, 0f)},
        {(0, 2), new Condition("s_shaped_long", -2.13f, 0f)},

        {(1, 0), new Condition("s_shaped", 2.77f, 180f)},
        {(1, 1), new Condition("vertical", 0f, 0f)},
        {(1, 2), new Condition("s_shaped", -2.77f, 0f)},

        {(2, 0), new Condition("s_shaped_long", 2.13f, 180f)},
        {(2, 1), new Condition("s_shaped", 0.66f, 180f)},
        {(2, 2), new Condition("vertical", -2.1f, 0f)},
    };

    private Dictionary<string, GameObject> tubes;

    // Keeps track of expt stats
    private Dictionary<string, string> stats;

    // Start is called before the first frame update
    void Start()
    {   
        Random.InitState(seed);
        tubes = new Dictionary<string, GameObject>(){
            {"vertical", VTube}, 
            {"s_shaped", STube}, 
            {"s_shaped_long", STubeLong}};
        stats = new Dictionary<string, string>();
        generateTube();
        for (int i=0; i<apples.Count; i++){
            apples[i].GetComponent<Rigidbody>().useGravity = false;
        }
        destVector = newDestVector();
    }

    // Runs every frame
    void Update()
    {
        GameObject apple = apples[0];
        if (curAppleIdx < apples.Count){
            apple = apples[curAppleIdx];
            float step = movingSpeed * Time.deltaTime;
            apple.transform.position = Vector3.MoveTowards(apple.transform.position, destVector, step);
            if (apple.transform.position == destVector){
                apple.GetComponent<Rigidbody>().useGravity = true;
                curAppleIdx += 1;
                destVector = newDestVector();
            }
        }
        else{
            int waitTime = (int) Mathf.Round(7 / Mathf.Log(movingSpeed + 1, 2));
            StartCoroutine(endExpt(waitTime));
        }
    }
    
    // Defines the ragnes of the apples basedo n the selected tube.
    void setApplesRanges(int state){
        float zlambda = 0.2f;
        float zValue = exptStates[(state, state)].tubeZPos;
        coordinateRanges["zmin"] = zValue - zlambda;
        coordinateRanges["zmax"] = zValue + zlambda;
    }

    // Generates one of the tubes (either vertical or s-shaped)
    void generateTube(){
        if (tubes.Count == 0 || numReceptacles < 1){
            return;
        }

        int dropLocation = Random.Range(0, numReceptacles);
        int finalLocation = Random.Range(0, numReceptacles);
        Condition exptCondition = exptStates[(dropLocation, finalLocation)];

        // Log expt parameters
        stats.Add("tube_type", exptCondition.tubeType);
        stats.Add("drop_location", dropLocation.ToString());
        stats.Add("final_location", finalLocation.ToString());

        // Set apples movement trajectory
        setApplesRanges(dropLocation);

        GameObject tube = tubes[exptCondition.tubeType];
        foreach (var item in tubes){
            string tubeType = item.Key;
            GameObject tubeObject = item.Value;

            if (tubeType == exptCondition.tubeType){
                tubeObject.SetActive(true);
                Vector3 tubePos = tubeObject.transform.position;
                tubeObject.transform.position = new Vector3(tubePos.x, tubePos.y, exptCondition.tubeZPos);
                tubeObject.transform.Rotate(0f, exptCondition.tubeYRot, 0f, Space.World);
            }
            else{
                tubeObject.SetActive(false);
            }
        }
    }

    // Generates a random destination vecotr for rewards.
    Vector3 newDestVector(){
        float x = Random.Range(coordinateRanges["xmin"], coordinateRanges["xmax"]);
        float y = Random.Range(coordinateRanges["ymin"], coordinateRanges["ymax"]);
        float z = Random.Range(coordinateRanges["zmin"], coordinateRanges["zmax"]);
        return new Vector3(x, y, z);
    }

    // Waits for 5 seconds and then ends the simulation.
    IEnumerator endExpt(int waitTime)
    {
        yield return new WaitForSeconds(waitTime);
        recorder.isRecording = false;
        GetComponent<ExptStatsWriter>().WriteYAML();
        Application.Quit();
    }

    // Returns a copy of the stats dictionary
    public Dictionary<string, string> GetStats(){
        return new Dictionary<string, string>(stats);
    }

}

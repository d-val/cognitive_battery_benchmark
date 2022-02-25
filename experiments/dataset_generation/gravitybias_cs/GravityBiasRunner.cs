using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GravityBiasRunner : MonoBehaviour
{

    public List<GameObject> tubes;
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

    private Dictionary<int, float> VTubeStates = new Dictionary<int, float>(){
        {0, 2.1f}, // left
        {1, 0f}, // middle
        {2, -2.1f}, // right
    };

    private Dictionary<int, float> STubeStates = new Dictionary<int, float>(){
        {0, -0.57f}, // left
        {1, -2.77f}, // middle
        {2, 0.52f}, // right
    };

    Dictionary<string, string> stats;

    // Start is called before the first frame update
    void Start()
    {   
        Random.InitState(seed);
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
            StartCoroutine(endExpt(5));
        }
    }
    
    // Defines the ragnes of the apples basedo n the selected tube.
    void setApplesRanges(int state){
        float zlambda = 0.2f;
        coordinateRanges["zmin"] = VTubeStates[state] - zlambda;
        coordinateRanges["zmax"] = VTubeStates[state] + zlambda;
    }

    // Generates one of the tubes (either vertical or s-shaped)
    void generateTube(){
        if (tubes.Count == 0){
            return;
        }
        int tubeIdx = Random.Range(0, tubes.Count);
        for (int i=0; i<tubes.Count; i++){
            if (i == tubeIdx){
                tubes[i].SetActive(true);
                
                if (i == 1){ // Vertical Tube
                    int dropLocation = Random.Range(0, VTubeStates.Count);
                    stats.Add("tube_type", "vertical");
                    stats.Add("drop_location", dropLocation.ToString());
                    stats.Add("final_location", dropLocation.ToString());
                    setApplesRanges(dropLocation);
                    var tubePos = tubes[i].transform.position;
                    tubes[i].transform.position = new Vector3(tubePos.x, tubePos.y, VTubeStates[dropLocation]);
                }
                else{ // S-Tube
                    int dropLocation = Random.Range(0, STubeStates.Count);
                    stats.Add("tube_type", "s_shaped");
                    stats.Add("drop_location", dropLocation.ToString());
                    setApplesRanges(dropLocation);
                    var tubePos = tubes[i].transform.position;
                    tubes[i].transform.position = new Vector3(tubePos.x, tubePos.y, STubeStates[dropLocation]);
                    if (dropLocation == 2){
                        tubes[i].transform.Rotate(0.0f, 180.0f, 0.0f, Space.World);
                        stats.Add("final_location", (dropLocation-1).ToString());
                    }
                    else{
                        stats.Add("final_location", (dropLocation+1).ToString());
                    }
                }
            }
            else{
                tubes[i].SetActive(false);
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

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using SysRandom = System.Random;

public class GravityBiasRunner : MonoBehaviour
{
    // A tube type describing the tube offset and occupancy range.
    private struct Tube
    {
        public int offset;          // Difference between the reward final location and drop locaiton.
        public int[] occupies;      // The range of reseptacles occupied by the tube relative to the drop location.
        public string name;         // The name identifier corresponding to the tube type.
        public bool reverseY;       // Whether the tube is reversed (i.e. rotated 180 degrees in Y dircetion).

        public Tube(int offset, int[] occupies, string name, bool reverseY){
            this.offset = offset;
            this.occupies = occupies;
            this.name = name;
            this.reverseY = reverseY;
        }
    }

    // A tube state describing the tube location.
    private struct TubeState
    {
        public Tube tube;         // Difference between the reward final location and drop locaiton.
        public int drop;          // The range of reseptacles occupied by the tube relative to the drop location.
        public int final;         // A name identifier for the tube type.

        public TubeState(Tube tube, int drop, int final){
            this.tube = tube;
            this.drop = drop;
            this.final = final;
        }
    }
    
    // Meshes for tubes
    public GameObject VTube;
    public GameObject STube;
    public GameObject STubeLong;
    public GameObject DoubleSTubeShort;
    public GameObject DoubleSTubeMedium;
    public GameObject DoubleSTubeLong;

    // List of tubes generated in the scene.
    public List<GameObject> tubes;

    // Main camera, used for viewing (Main Camera)
    public GameObject sceneCamera;

    // Recorder, used to start/stop recording
    public FrameRecorder recorder;

    // Number of receptacles in scene, choose at random if -1
    public int numReceptacles = -1;

    // Number of tubes in scene, choose at random if -1
    public int numTubes = -1;

    // Choice of receptacle type (e.g. cup, bowl, etc..)
    public int receptType = -1; 
    public List<GameObject> receptaclePrefabs;

    // List of bowls generated in the scene.
    public List<GameObject> receptacles = new List<GameObject>();

    // Reward randomization?
    public int numRewards = -1; // how many there are
    public int rewardType = -1; // apple, tomato, etc..
    public List<GameObject> rewardPrefabs;

    // List of rewards generated in the scene.
    public List<GameObject> rewards = new List<GameObject>();

    // How fast you want rewards to move
    public int movingSpeed;
    
    // Randomization seed
    public int seed;
    
    // X and Y positions that rewards need to end up in.
    public Dictionary<string, float> coordinateRanges = new Dictionary<string, float>(){
        {"xmin", -2f},
        {"xmax", -1.9f},

        {"ymin", 15.5f},
        {"ymax", 16.25f},

        {"zmin", 0},
        {"zmax", 0},
    };


    private int curRewardIdx = 0;
    private Vector3 destVector;

    private Dictionary<string, GameObject> tubesPrefabs;

    // Keeps track of expt stats
    private Dictionary<string, string> stats;

    // Start is called before the first frame update
    void Start()
    {   
        Random.InitState(seed);

        List<Tube> testTubes = new List<Tube>(){
            new Tube(0, new int[]{0,0}, "vertical", false),

            new Tube(1, new int[]{0,1}, "s_shaped", false),
            new Tube(-1, new int[]{-1,0}, "s_shaped", true),

            new Tube(2, new int[]{0,2}, "s_shaped_long", false),
            new Tube(-2, new int[]{-2,0}, "s_shaped_long", true),

            new Tube(0, new int[]{-1,0}, "double_s_shaped_short", false),
            new Tube(0, new int[]{0,1}, "double_s_shaped_short", true),

            new Tube(1, new int[]{-1,1}, "double_s_shaped_medium", false),
            new Tube(-1, new int[]{-1,1}, "double_s_shaped_medium", true),

            new Tube(0, new int[]{-2,0}, "double_s_shaped_long", false),
            new Tube(0, new int[]{0,2}, "double_s_shaped_long", true),
        };

        tubesPrefabs = new Dictionary<string, GameObject>(){
            {"vertical", VTube}, 
            {"s_shaped", STube}, 
            {"s_shaped_long", STubeLong},
            {"double_s_shaped_short", DoubleSTubeShort},
            {"double_s_shaped_medium", DoubleSTubeMedium},
            {"double_s_shaped_long", DoubleSTubeLong},
        };

        stats = new Dictionary<string, string>();

        // Generating rewards.
        if (numRewards == -1){ // unspecified
            numRewards = Random.Range(3, 6);
        }
        if (rewardType == -1){ // unspecified
            rewardType = Random.Range(0, rewardPrefabs.Count);
        }
        for (int i=0; i<numRewards; i++){
            Vector3 rewardPos = new Vector3(Random.Range(2.33f, 3f), 9f, Random.Range(-1.8f, -0.67f));
            GameObject reward = Instantiate(rewardPrefabs[rewardType], rewardPos, Quaternion.identity);
            rewards.Add(reward);
        }

        // Generating receptacles.
        if (numReceptacles == -1){ // unspecified
            numReceptacles = Random.Range(2, 9);
        }
        if (numTubes == -1){ // unspecified
            numTubes = Random.Range(1, numReceptacles);
        }
        if (receptType == -1){ // unspecified
            receptType = 0;
        }

        // Places receptacles on table
        float receptStartPos = (numReceptacles*1.35f)/2 - 1f;
        for (int i=0; i<numReceptacles; i++){
            Vector3 receptPos = new Vector3(-2f, 8.47f, receptStartPos - i*1.35f);
            GameObject recept = Instantiate(receptaclePrefabs[receptType], receptPos, Quaternion.identity);
            receptacles.Add(recept);
        }

        // Determining reward drop and final locations
        int margin = Mathf.Min(2, numReceptacles - numTubes); // determines by how far the drop and final locations should be sepearated
        int dropLocation = Random.Range(0, numReceptacles);
        int finalLocation = Random.Range(Mathf.Max(dropLocation-margin, 0) , Mathf.Min(dropLocation+margin + 1, numReceptacles));
        List<TubeState> condition = generateCondition(testTubes, dropLocation, finalLocation);
        foreach (TubeState ts in condition){
            PlaceTube(ts);
        }

        setRewardsRanges(dropLocation);

        stats.Add("drop_location", dropLocation.ToString());
        stats.Add("final_location", finalLocation.ToString());
        stats.Add("reward_type", rewardPrefabs[rewardType].name);
        stats.Add("num_receptacles", numReceptacles.ToString());
        stats.Add("num_tubes", numTubes.ToString());
        stats.Add("num_rewards", numRewards.ToString());

        destVector = newDestVector();
        Debug.Log("numReceptacles: " + numReceptacles.ToString());
        Debug.Log("numTubes: " + numTubes.ToString());
        Debug.Log("numRewards: " + numRewards.ToString());
    }

    // Runs every frame
    void Update()
    {
        GameObject reward = rewards[0];
        if (curRewardIdx < rewards.Count){
            reward = rewards[curRewardIdx];
            reward.GetComponent<Rigidbody>().useGravity = false;
            reward.GetComponent<Rigidbody>().isKinematic = true;
            float step = movingSpeed * Time.deltaTime;
            reward.transform.position = Vector3.MoveTowards(reward.transform.position, destVector, step);
            if (reward.transform.position == destVector){
                reward.GetComponent<Rigidbody>().useGravity = true;
                reward.GetComponent<Rigidbody>().isKinematic = false;
                curRewardIdx += 1;
                destVector = newDestVector();
            }
        }
        else{
            // Wait for some time then stop recording
            int waitTime = (int) Mathf.Round(7 / Mathf.Log(movingSpeed + 1, 2));
            StartCoroutine(endExpt(waitTime));
        }
        
    }
    
    // Defines the ragnes of the rewards based on the selected tube.
    void setRewardsRanges(int state){
        float zValue = receptacles[state].transform.position.z;
        coordinateRanges["zmin"] = zValue;
        coordinateRanges["zmax"] = zValue;
    }

    // Generates one of the tubes (either vertical or s-shaped)
    List<TubeState> generateCondition(List<Tube> tubes, int drop, int final){
        if (!(tubes.Count > 0 && Mathf.Abs(drop - final) <= 2 && drop < numReceptacles && final < numReceptacles)){
            Debug.Log(tubes.Count > 0);
            Debug.Log(Mathf.Abs(drop - final) <= 2);
            Debug.Log(drop < numReceptacles);
            Debug.Log(final < numReceptacles);
            return new List<TubeState>();
        }
        SysRandom randomizer = new SysRandom(seed);

        bool validTube(Tube tube, int drop){
            return (drop + tube.occupies[0] >= 0) && (drop + tube.occupies[1] < numReceptacles);
        }

        List<TubeState> finalPlacements = new List<TubeState>();
        bool[] occupied = new bool[numReceptacles];
        tubes = tubes.OrderBy(x => randomizer.Next()).ToList();

        int margin = numReceptacles - numTubes;
        int tOccupDiff;
        
        foreach (Tube t in tubes){
            tOccupDiff = t.occupies[1] - t.occupies[0];
            if ((t.offset == final - drop) && validTube(t, drop) && (margin >= tOccupDiff)){
                // Main Tube
                finalPlacements.Add(new TubeState(t, drop, final));
                margin -= tOccupDiff;
                // Marking its range as occupied
                for (int i=t.occupies[0]; i<=t.occupies[1]; i++){
                    occupied[drop+i] = true;
                }
                break;
            }
        }
        if (finalPlacements.Count == 0){
            Debug.Log("SNAFUUUUU!!!");
            // throw new Exception("No tube combination matches initial conditions of numTubes and numReceptacles");
        }

        while (occupied.Contains(false) && finalPlacements.Count < numTubes){
            tubes = tubes.OrderBy(x => randomizer.Next()).ToList();
            int[] indices = Enumerable.Range(0, numReceptacles).OrderBy(x => randomizer.Next()).ToArray();
            foreach (int i in indices){
                if (occupied[i]){continue;}
                if (finalPlacements.Count >= numTubes){break;}
                foreach (Tube t in tubes){
                    if (!validTube(t, i)){continue;}

                    tOccupDiff = t.occupies[1] - t.occupies[0];
                    if (!(margin >= tOccupDiff)){
                        continue;
                    }

                    bool empty = true;
                    for (int j=t.occupies[0]; j<=t.occupies[1]; j++){
                        if (occupied[i+j]==true){empty = false;}
                    }
                    if (empty){
                        for (int j=t.occupies[0]; j<=t.occupies[1]; j++){
                            occupied[i+j] = true;
                        }
                        finalPlacements.Add(new TubeState(t, i, i+t.offset));
                        margin -= tOccupDiff;
                        break;
                    }
                }
            }
        }
        return finalPlacements;
    }

    private void PlaceTube(TubeState state){

        // Creating tube with default transform.
        GameObject tubeInstance = Instantiate(tubesPrefabs[state.tube.name]);
        tubes.Add(tubeInstance);

        // Getting the Z-coordinate of the tube.   
        float zPos = 0f; 
        if (state.tube.name == "s_shaped"){
            // Special case, needs offsetting
            zPos = receptacles[state.final].transform.position.z + (state.tube.reverseY ? .6f : -.6f);
        }
        else{
            zPos = receptacles[state.final].transform.position.z;
        }

        Vector3 tubePos = tubeInstance.transform.position;
        tubeInstance.transform.position = new Vector3(tubePos.x, tubePos.y, zPos);

        // Fixing the Y-rotation of the tube. 
        if (state.tube.reverseY){
            tubeInstance.transform.Rotate(0f, 180f, 0f, Space.World);
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
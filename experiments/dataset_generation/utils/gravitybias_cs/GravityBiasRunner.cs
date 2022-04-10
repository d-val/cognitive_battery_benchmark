using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GravityBiasRunner : MonoBehaviour
{

    public GameObject VTube;
    public GameObject STube;
    public GameObject STubeLong;
    public GameObject DoubleSTubeShort;
    public GameObject DoubleSTubeMedium;
    public GameObject DoubleSTubeLong;

    public int numReceptacles;

    public GameObject sceneCamera;
    public FrameRecorder recorder;

    public int numRewards = 4;
    public List<GameObject> rewardPrefabs;
    public int rewardType; 

    private List<GameObject> rewards = new List<GameObject>();

    public int movingSpeed;
    public float targetYPos;
    
    public int seed;
    
    public Dictionary<string, float> coordinateRanges = new Dictionary<string, float>(){
        {"xmin", -2f},
        {"xmax", -1.9f},

        {"ymin", 15.5f},
        {"ymax", 16.5f},

        {"zmin", 0},
        {"zmax", 0},
    };

    private int curRewardIdx = 0;
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

    // Maps each possible drop/final pair into a list of lists of game conditions. The first list hierarchy denotes all the possible conditions for the drop/final pair.
    //  the second list hierarchy denotes the tubes and their positions/rotations within that hierarchy.
    // Total: 33 conditions
    private Dictionary<(int drop, int final), List<List<Condition>>> exptStates = new Dictionary<(int, int), List<List<Condition>>>(){
        {// 8 conditions
            (0, 0), 
            new List<List<Condition>>(){
                    new List<Condition>(){
                        new Condition("vertical", 2.1f, 0f),
                    },
                    new List<Condition>(){
                        new Condition("vertical", 2.1f, 0f),
                        new Condition("s_shaped", -2.77f, 0f),
                    },
                    new List<Condition>(){
                        new Condition("vertical", 2.1f, 0f),
                        new Condition("s_shaped", 0.66f, 180f)
                    },
                    new List<Condition>(){
                        new Condition("vertical", 2.1f, 0f),
                        new Condition("double_s_shaped_short", 0f, 180f),
                    },
                    new List<Condition>(){
                        new Condition("vertical", 2.1f, 0f),
                        new Condition("double_s_shaped_short", -2.1f, 0f),
                    },
                    new List<Condition>(){
                        new Condition("double_s_shaped_long", 2.1f, 180f),
                    },
                    new List<Condition>(){
                        new Condition("double_s_shaped_short", 2.1f, 180f),
                    },
                    new List<Condition>(){
                        new Condition("double_s_shaped_short", 2.1f, 180f),
                        new Condition("vertical", -2.1f, 0f),
                    },
                }
        },
        {// 2 conditions 
            (0, 1),
            new List<List<Condition>>(){
                new List<Condition>(){
                     new Condition("s_shaped", -0.57f, 0f),
                },
                new List<Condition>(){
                     new Condition("s_shaped", -0.57f, 0f),
                     new Condition("vertical", -2.1f, 0f),
                },
            }
        },
        {// 1 conditions
            (0, 2),
            new List<List<Condition>>(){
                new List<Condition>(){
                    new Condition("s_shaped_long", -2.13f, 0f)
                }
            }
        },
        {// 3 conditions
            (1, 0),
            new List<List<Condition>>(){
                new List<Condition>(){
                    new Condition("s_shaped", 2.77f, 180f),
                },
                new List<Condition>(){
                    new Condition("s_shaped", 2.77f, 180f),
                    new Condition("vertical", -2.1f, 0f),
                },
                new List<Condition>(){
                    new Condition("double_s_shaped_medium", 2.13f, 180f),
                },
            }
        },        
        {// 5 conditions
            (1, 1),
            new List<List<Condition>>(){
                new List<Condition>(){
                    new Condition("vertical", 0f, 0f),
                },
                new List<Condition>(){
                    new Condition("double_s_shaped_short", 0f, 0f),
                },
                new List<Condition>(){
                    new Condition("double_s_shaped_short", 0f, 0f),
                    new Condition("vertical", -2.1f, 0f),
                },
                new List<Condition>(){
                    new Condition("double_s_shaped_short", 0f, 180f),
                },
                new List<Condition>(){
                    new Condition("double_s_shaped_short", 0f, 180f),
                    new Condition("vertical", 2.1f, 0f),
                },
            }
        },   
        {// 3 conditions
            (1, 2),
            new List<List<Condition>>(){
                new List<Condition>(){
                    new Condition("s_shaped", -2.77f, 0f),
                },
                new List<Condition>(){
                    new Condition("s_shaped", -2.77f, 0f),
                    new Condition("vertical", 2.1f, 0f),
                },
                new List<Condition>(){
                    new Condition("double_s_shaped_medium", -2.13f, 0f),
                },
            }
        },  
        {// 1 conditions
            (2, 0),
            new List<List<Condition>>(){
                new List<Condition>(){
                    new Condition("s_shaped_long", 2.13f, 180f)
                },
            }
        },
        {// 2 conditions
            (2, 1),
            new List<List<Condition>>(){
                new List<Condition>(){
                    new Condition("s_shaped", 0.66f, 180f),
                },
                new List<Condition>(){
                    new Condition("s_shaped", 0.66f, 180f),
                    new Condition("vertical", 2.1f, 0f),
                },
            }
        }, 
        {// 8 conditions
            (2, 2),
            new List<List<Condition>>(){
                new List<Condition>(){
                    new Condition("vertical", -2.1f, 0f),
                },
                new List<Condition>(){
                    new Condition("vertical", -2.1f, 0f),
                    new Condition("double_s_shaped_short", 2.1f, 180f),
                },
                new List<Condition>(){
                    new Condition("vertical", -2.1f, 0f),
                    new Condition("double_s_shaped_short", 0f, 0f),
                },
                new List<Condition>(){
                    new Condition("vertical", -2.1f, 0f),
                    new Condition("s_shaped", -0.57f, 0f),
                },
                new List<Condition>(){
                    new Condition("vertical", -2.1f, 0f),
                    new Condition("s_shaped", 2.77f, 180f),
                },
                new List<Condition>(){
                    new Condition("double_s_shaped_long", -2.1f, 0f),
                },
                new List<Condition>(){
                    new Condition("double_s_shaped_short", -2.1f, 0f),
                },
                new List<Condition>(){
                    new Condition("double_s_shaped_short", -2.1f, 0f),
                    new Condition("vertical", 2.1f, 0f),
                },
            }
        },
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
            {"s_shaped_long", STubeLong},
            {"double_s_shaped_short", DoubleSTubeShort},
            {"double_s_shaped_medium", DoubleSTubeMedium},
            {"double_s_shaped_long", DoubleSTubeLong},
        };

        for (int i=0; i<numRewards; i++){
            Vector3 rewardPos = new Vector3(Random.Range(2.33f, 3f), 9f, Random.Range(-1.8f, -0.67f));
            GameObject reward = Instantiate(rewardPrefabs[rewardType], rewardPos, Quaternion.identity);
            rewards.Add(reward);
        }

        stats = new Dictionary<string, string>();
        stats.Add("reward_type", rewardPrefabs[rewardType].name);

        generateTube();
        destVector = newDestVector();
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
            int waitTime = (int) Mathf.Round(7 / Mathf.Log(movingSpeed + 1, 2));
            StartCoroutine(endExpt(waitTime));
        }
    }
    
    // Defines the ragnes of the rewards based on the selected tube.
    void setRewardsRanges(int state){
        float zlambda = 0f;
        float zValue = exptStates[(state, state)][0][0].tubeZPos;
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
        List<List<Condition>> exptConditions = exptStates[(dropLocation, finalLocation)];
        List<Condition> exptCondition = exptConditions[Random.Range(0, exptConditions.Count)];

        // Log expt parameters
        stats.Add("reward_tube_types", exptCondition[0].tubeType);
        stats.Add("drop_location", dropLocation.ToString());
        stats.Add("final_location", finalLocation.ToString());

        // Set rewards movement trajectory
        setRewardsRanges(dropLocation);

        foreach (Condition tubeState in exptCondition){
            GameObject tubeObject = tubes[tubeState.tubeType];
            tubeObject.SetActive(true);
            Vector3 tubePos = tubeObject.transform.position;
            tubeObject.transform.position = new Vector3(tubePos.x, tubePos.y, tubeState.tubeZPos);
            tubeObject.transform.Rotate(0f, tubeState.tubeYRot, 0f, Space.World);
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

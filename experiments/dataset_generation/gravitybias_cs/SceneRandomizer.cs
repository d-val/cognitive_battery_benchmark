using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SceneRandomizer : MonoBehaviour
{
    /*
    Randomizes the settings of the scene at initilization, including:
        - Lights
        - Tube color
        - Chairs colors
        - Walls colors
        - Bowls colors.
    */

    public int seed;
    public bool randomizeLights;
    public bool randomizeTubeColor;
    public bool randomizeChairsColors;
    public bool randomizeWallsColors;
    public bool randomizeBowlsColors;
    public bool randomizeFOV;

    public Light mainLight; // Main light in the scene.
    public GameObject STube;
    public GameObject straigthTube;
    public List<GameObject> chairs;
    public List<GameObject> walls;
    public List<GameObject> bowls;
    public Camera showCam;
    public Camera recordCam;
    public float fov = 55;
    
    Dictionary<string, string> stats;

    void Start() 
    {
        Random.InitState(seed);
        stats = new Dictionary<string, string>();
        stats.Add("seed", seed.ToString());

        // Randomize the light of the scene
        if (randomizeLights){
            Color32 newLightColor = generateRandomColor(160, 255);
            mainLight.color = newLightColor;
            stats.Add("light_color", newLightColor.ToString("F2"));
        }

        // Randomize the color of the tube
        if (randomizeTubeColor){
            Color32 newTubeColor = generateRandomColor(160, 255);
            STube.gameObject.GetComponent<Renderer>().material.color = newTubeColor;
            straigthTube.gameObject.GetComponent<Renderer>().material.color = newTubeColor;
            stats.Add("tube_color", newTubeColor.ToString("F2"));
        }

        // Randomize the color of the background chairs and furniture
        if (randomizeChairsColors){
            Color32 newChairColor;
            foreach (GameObject chair in chairs){
                newChairColor = generateRandomColor(160, 255);
                foreach(Renderer r in chair.GetComponentsInChildren<Renderer>()){
                    r.material.color = newChairColor;
                }
            }
        }

        // Randomize the color of the walls
        if (randomizeWallsColors){
            Color32 newWallColor = generateRandomColor(100, 255);
            foreach (GameObject wall in walls){
                foreach(Renderer r in wall.GetComponentsInChildren<Renderer>()){
                    r.material.color = newWallColor;
                }
            }
        }


        // Randomize the color of the bowls
        if (randomizeBowlsColors){
            Color32 newBowlColor = generateRandomColor(100, 255);
            foreach (GameObject bowl in bowls){
                foreach(Renderer r in bowl.GetComponentsInChildren<Renderer>()){
                    r.material.color = newBowlColor;
                }
            }
            stats.Add("receptacles_color", newBowlColor.ToString("F2"));
        }

        // Randomize the field of view of the camera
        if (randomizeFOV){
            fov = Random.Range(50.0f, 65.0f);
        }
        showCam.fieldOfView = fov;
        recordCam.fieldOfView = fov;
        stats.Add("fov", fov.ToString("F2"));
    }

    // Returns a new random color between (min, min, min) and (max, max, max) RGB
    Color32 generateRandomColor(int min, int max){
        return new Color32(System.Convert.ToByte(Random.Range(min, max)),
                        System.Convert.ToByte(Random.Range(min, max)),
                        System.Convert.ToByte(Random.Range(min, max)),
                        255); 
    }

    // Returns a copy of the stats dictionary
    public Dictionary<string, string> GetStats(){
        return new Dictionary<string, string>(stats);
    }
}

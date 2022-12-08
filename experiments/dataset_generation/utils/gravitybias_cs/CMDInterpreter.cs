using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class CMDInterpreter : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        string[] args = System.Environment.GetCommandLineArgs ();
        int screenWidth = 1920; // Default width
        int screenHeight = 1080; // Default height

        FrameRecorder fr = GetComponent<FrameRecorder>();

        for (int i = 0; i < args.Length; i++) {
            if (args[i] == "--record") {
                fr.isRecording = true;
            }

            if (args[i] == "--depth") {
                fr.isRecordingDepth = true;
            }

            if (args[i] == "--segmentation") {
                fr.isRecordingSegmentation = true;
            }

            // Enable dev logging, if flagged
            if (args[i] == "--dev"){
                GetComponent<DevLogger>().enabled = true;
            }

            // Set the output directory of frames, video, stats, etc ..
            if (args[i] == "--outdir"){
                // Initializes the output directory
                string outPath = "";
                if (i+1 < args.Length){
                    outPath = args[i+1];
                }

                // Defining and creating output directory if it does not exist.
                if (outPath == ""){
                    outPath = Directory.GetCurrentDirectory();
                }
                if (!Directory.Exists(outPath))
                {
                    Directory.CreateDirectory(outPath);
                }
                
                // Sets the recorder's directory
                GetComponent<FrameRecorder>().outPath = Path.Combine(outPath, "human_readable");
                GetComponent<ExptStatsWriter>().statsOutPath = outPath;
            }

            // Set the experiment randomizer seed, if provided.
            if (args[i] == "--seed"){
                // Initialize to a default seed of 0.
                int seed = 0;
                if (i+1 < args.Length){
                    int.TryParse(args[i+1], out seed);
                }

                // Sets the randomizer script seed.
                GetComponent<SceneRandomizer>().seed = seed;
                GetComponent<GravityBiasRunner>().seed = seed;
            }

            // Set the camera fov, if provided
            if (args[i] == "--fov"){
                float fov = 55;
                if (i+1 < args.Length){
                    if (float.TryParse(args[i+1], out fov)){
                        i++;
                    }
                }

                // Sets the randomizer script seed.
                GetComponent<SceneRandomizer>().randomizeFOV = false;
                GetComponent<SceneRandomizer>().fov = fov;
            }

            // Set the playing speed, if provided
            if (args[i] == "--speed"){
                int speed = 3;
                if (i+1 < args.Length){
                    if (int.TryParse(args[i+1], out speed)){
                        i++;
                    }
                }

                // Sets the randomizer script seed.
                GetComponent<GravityBiasRunner>().movingSpeed = speed;
            }

            // Set screen width and height if provided
            if (args[i] == "--width"){
                if (i+1 < args.Length){
                    int.TryParse(args[i+1], out screenWidth);
                }
            }

            if (args[i] == "--height"){
                if (i+1 < args.Length){
                    int.TryParse(args[i+1], out screenHeight);
                }
            }

            // Sets the reward type if provided
            if (args[i] == "--reward"){
                if (i+1 < args.Length){
                    int rewardType = 0;
                    int.TryParse(args[i+1], out rewardType);
                    GetComponent<GravityBiasRunner>().rewardType = rewardType;
                }
            }

            // Sets the number of rewards if provided
            if (args[i] == "--num-rewards"){
                if (i+1 < args.Length){
                    int numRewards = 0;
                    int.TryParse(args[i+1], out numRewards);
                    GetComponent<GravityBiasRunner>().numRewards = numRewards;
                }
            }

            // Sets the number of receptacles if provided
            if (args[i] == "--num-receptacles"){
                if (i+1 < args.Length){
                    int numReceptacles = 0;
                    int.TryParse(args[i+1], out numReceptacles);
                    GetComponent<GravityBiasRunner>().numReceptacles = numReceptacles;
                }
            }

            // Sets the number of tubes if provided
            if (args[i] == "--num-tubes"){
                if (i+1 < args.Length){
                    int numTubes = 0;
                    int.TryParse(args[i+1], out numTubes);
                    GetComponent<GravityBiasRunner>().numTubes = numTubes;
                }
            }
        }
        Screen.SetResolution(screenWidth, screenHeight, false);
    }
}
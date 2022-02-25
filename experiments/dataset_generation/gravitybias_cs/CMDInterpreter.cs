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

        for (int i = 0; i < args.Length; i++) {
            if (args[i] == "-record") {
                // Start recorder
                GetComponent<FrameRecorder>().isRecording = true;
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
                GetComponent<FrameRecorder>().outPath = Path.Combine(outPath, "human_readable", "frames");
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
        }
        Screen.SetResolution(screenWidth, screenHeight, false);
    }
}
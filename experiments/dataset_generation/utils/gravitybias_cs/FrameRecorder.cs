using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class FrameRecorder : MonoBehaviour
{

    /*
    Saves snapshots of the played video every captureRate frames at the path outPath/captures.
    */

    public Camera recordingCamera;          // camera used for recording
    public int capturRate;                  // the number of frames between every capture
    public string outPath;                  // path at which captures are saved
    public bool isRecording;                // A switch to turn on/off recording.
    public bool isRecordingDepth;           // A switch to turn on/off depth recording.
    public bool isRecordingSegmentation;    // A switch to turn on/off segmentation recording.
    
    private int capturesCounter = 0;
    private int frame = 0;

    void Capture(){
        /*
        Saves the current camera screen into the output path.
        */
        
        string framesCapturePath = Path.Combine(outPath, "frames");
        recordingCamera.GetComponent<ImageSynthesis>().Save("frame" + "_" + capturesCounter.ToString() + ".jpeg", -1, -1, framesCapturePath, "_img");
        
        if (isRecordingDepth){
            string depthCapturePath = Path.Combine(outPath, "depth");
            Debug.Log("Saving depth at: " + depthCapturePath + "/frame" + "_" + capturesCounter.ToString() + ".jpeg");
            recordingCamera.GetComponent<ImageSynthesis>().Save("frame" + "_" + capturesCounter.ToString() + ".jpeg", -1, -1, depthCapturePath, "_depth");
        }

        if (isRecordingSegmentation){
            string segmentedCapturePath = Path.Combine(outPath, "segmented");
            Debug.Log("Saving segmented at: " + segmentedCapturePath + "/frame" + "_" + capturesCounter.ToString() + ".jpeg");
            recordingCamera.GetComponent<ImageSynthesis>().Save("frame" + "_" + capturesCounter.ToString() + ".jpeg", -1, -1, segmentedCapturePath, "_id");
        }
    }

    void Update(){
        /*
        Captures the current frame each captureRate frames.
        */

        if (isRecording && frame % capturRate == 0){
            Capture();
            capturesCounter++;
        }
        frame++;
    }
}

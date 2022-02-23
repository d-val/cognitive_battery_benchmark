using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class FrameRecorder : MonoBehaviour
{

    /*
    Saves snapshots of the played video every captureRate frames at the path outPath/captures.
    */

    public Camera recordingCamera;  // camera used for recording
    public int capturRate;          // the number of frames between every capture
    public string outPath;          // path at which captures are saved
    public bool isRecording;        // A switch to turn on/off recording.
    
    private int capturesCounter = 0;
    private int frame = 0;

    void Start(){
        recordingCamera.targetTexture = new RenderTexture(Screen.width, Screen.height, 24);
    }

    void Capture(){
        /*
        Saves the current camera screen into the output path.
        */
        Texture2D tex = RTImage();
        byte[] bytes;
        bytes = tex.EncodeToPNG();
        string capturePath = Path.Combine(outPath, "frame" + "_" + capturesCounter.ToString() + ".png");
        File.WriteAllBytes(capturePath, bytes);
    }

    Texture2D RTImage()
    {
        /*
        Reads the pixels of the camera and extracts them into a texture.
        */

        var currentRT = RenderTexture.active;
        RenderTexture.active = recordingCamera.targetTexture;

        // Render the camera's view.
        recordingCamera.Render();

        // Make a new texture and read the active Render Texture into it.
        Texture2D image = new Texture2D(recordingCamera.targetTexture.width, recordingCamera.targetTexture.height);
        image.ReadPixels(new Rect(0, 0, recordingCamera.targetTexture.width, recordingCamera.targetTexture.height), 0, 0);
        image.Apply();

        // Replace the original active Render Texture.
        RenderTexture.active = currentRT;
        return image;
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

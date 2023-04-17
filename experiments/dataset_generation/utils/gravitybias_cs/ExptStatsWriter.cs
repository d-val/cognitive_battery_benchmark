using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class ExptStatsWriter : MonoBehaviour
{
    SceneRandomizer randomizer;
    GravityBiasRunner expt_runner;
    Camera recordingCamera;

    Dictionary<string, string> stats;

    public string statsOutPath;

    void Start(){
        randomizer = GetComponent<SceneRandomizer>();
        expt_runner = GetComponent<GravityBiasRunner>();
        stats = new Dictionary<string, string>();

        // Constant stats
        stats.Add("reward_type", "apple"); // For now, only apples.
        stats.Add("receptacle_type", "bowl"); // For now, only bowls.
    }
    
    void UpdaetStats(){
        foreach (var item in randomizer.GetStats()){
            if (!stats.ContainsKey(item.Key)){
                stats.Add(item.Key, item.Value);
            }
        }

        foreach (var item in expt_runner.GetStats()){
            if (!stats.ContainsKey(item.Key)){
                stats.Add(item.Key, item.Value);
            }
        }
    }
    
    public void WriteYAML()
    {
        UpdaetStats();
        if (statsOutPath == ""){
            statsOutPath = "./";
        }

        string yamlFileName = "experiment_stats.yaml";
        string hrDir = Path.Combine(statsOutPath, "human_readable");
        string mrDir = Path.Combine(statsOutPath, "machine_readable");
        if (!Directory.Exists(hrDir)){
            Directory.CreateDirectory(hrDir);
        }
        if (!Directory.Exists(mrDir)){
            Directory.CreateDirectory(mrDir);
        }
        
        StreamWriter hrFile = File.CreateText(Path.Combine(hrDir, yamlFileName));
        StreamWriter mrFile = File.CreateText(Path.Combine(mrDir, yamlFileName));
        foreach (var item in stats){
            hrFile.WriteLine ("{0}: {1}", item.Key, item.Value);
            mrFile.WriteLine ("{0}: {1}", item.Key, item.Value);
        }
        hrFile.Close();
        mrFile.Close();
    }
}

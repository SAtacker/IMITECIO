using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class numberScript : MonoBehaviour
{
    public static float vertVel = 0;
    public static float timeTotal = 0;
    public static int coinTotal = 0; 

    public static string lvlCompStatus = "";

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        timeTotal += Time.deltaTime;
        
        if(lvlCompStatus == "fail")
        {
            SceneManager.LoadScene("Scoreboard"); 
        }
        if(lvlCompStatus == "success")
        {
            SceneManager.LoadScene("Scoreboard"); 
        }

    }
}

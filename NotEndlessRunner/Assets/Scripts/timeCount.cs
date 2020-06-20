using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class timeCount : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        //This should work but Unity decided to be an ass.
        GetComponent<TextMesh>().text = "Time: " + numberScript.timeTotal;
    }
}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class CharacterMovementScript : MonoBehaviour
{
    private Animator animator;
    private int lane;
    public Transform explodeObj;

    // Start is called before the first frame update
    void Start()
    {
        lane = 0;
        animator = GetComponent<Animator>(); 
    }

    // Update is called once per frame
    void Update()
    {
        if(Input.GetKeyDown(KeyCode.LeftArrow))
        {
            if (lane > -1)
                lane--; 
        }
        if(Input.GetKeyDown(KeyCode.RightArrow))
        {
            if (lane < 1)
                lane++;
        }
        if(Input.GetKeyDown(KeyCode.Space))
        {
            animator.SetTrigger("JumpButtonDown");
            animator.SetTrigger("continueRunning"); 
        }
        Vector3 newPosition = transform.position;
        newPosition.x = lane;
        transform.position = newPosition;
        transform.Rotate(Vector3.up, 0.0f); 
    }

    void OnCollisionEnter(Collision collided)
    {
        if(collided.gameObject.tag == "Obstruction")
        {
            numberScript.lvlCompStatus = "fail";
            Instantiate(explodeObj, transform.position, explodeObj.rotation); 
            Destroy(gameObject);
        }
        if(collided.gameObject.tag == "EndTheGame")
        {

            numberScript.lvlCompStatus = "success"; 
        }
    }
}

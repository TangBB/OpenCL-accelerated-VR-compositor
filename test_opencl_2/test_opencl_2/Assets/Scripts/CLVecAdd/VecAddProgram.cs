using UnityEngine;
using Cloo;
using System;
using System.IO;
using System.Collections;

namespace GodArs
{
	namespace CL
	{
		public class VecAddProgram : MonoBehaviour 
		{
			ComputePlatform platform        = null;
			ComputeDevice device                = null;
			ComputeContext context                = null;
			VecAdd pgVecAdd                                = null;
			StringWriter log                        = null;

			// Use this for initialization
			void Start () 
			{
				try
				{
					// Choose Platform
					// In my mac, platform: Apple device: Iris
					platform        = ComputePlatform.Platforms[0];
					device                = platform.Devices[1];

					Debug.Log (string.Format ("Platform: {0} Device: {1}", platform.Name, device.Name)); 

					// Get Context
					ComputeContextPropertyList properties = new ComputeContextPropertyList (platform);
					context = new ComputeContext (ComputeDeviceTypes.Gpu, properties, null, IntPtr.Zero);
					log = new StringWriter ();
					pgVecAdd = new VecAdd ();
				}
				catch (Exception e)
				{
					Debug.Log (e.ToString ());
				}
			}

			void RunProgram ()
			{
				// Execute Vector Add cl program
				pgVecAdd.Run (context, log);
			}

			// Update is called once per frame
			void Update () 
			{
				if (Input.GetKeyDown (KeyCode.Return))
				{
					Debug.Log ("Run VecAdd");
					RunProgram ();
				}
			}
		}
	}
}
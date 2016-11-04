using UnityEngine;
using Cloo;
using System;
using System.IO;
using System.Collections;

namespace GodArs
{
	namespace CL
	{
		public class VecAdd
		{
			ComputeProgram program;

			string clProgramSource;

			public string Name
			{
				get { return "VecAdd"; }
			}

			public string Description
			{
				get { return "CL Vector Addition"; }
			}

			private string path;

			public void Run (ComputeContext context, TextWriter log)
			{
				// Create host part data
				int count = 1024;
				float[] h_A = new float[count];
				float[] h_B = new float[count];
				float[] h_C = new float[count];

				// Init data
				System.Random rand = new System.Random (1);
				for (int i = 0; i < count; ++i)
				{
					h_A[i] = (float)(rand.NextDouble () * 100);
					h_B[i] = (float)(rand.NextDouble () * 100);
				}

				path = Environment.CurrentDirectory + "/Assets/Scripts/CLVecAdd/res";

				// Write To File
				StreamWriter swBefore = new StreamWriter (path + "/exBefore.txt");
				for (int i = 0; i < count; ++i)
				{
					swBefore.WriteLine ("{0} - {1} - {2}", h_A[i], h_B[i], h_C[i]);
				}

				// Create Input Buffer
				ComputeBuffer<float> d_A = new ComputeBuffer<float> (context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, h_A);
				ComputeBuffer<float> d_B = new ComputeBuffer<float> (context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, h_B);

				// Create Output BUffer
				ComputeBuffer<float> d_C = new ComputeBuffer<float> (context, ComputeMemoryFlags.WriteOnly, h_C.Length);

				// Load Program Source
				StreamReader srProgram = new StreamReader (Environment.CurrentDirectory + "/Assets/Scripts/CLVecAdd/kernel.cl");
				clProgramSource = srProgram.ReadToEnd ();

				// Create & Build the opencl program
				program = new ComputeProgram (context, clProgramSource);
				program.Build (null, null, null, IntPtr.Zero);

				// Create the kernel
				ComputeKernel kernel = program.CreateKernel ("VectorAdd");
				kernel.SetMemoryArgument (0, d_A);
				kernel.SetMemoryArgument (1, d_B);
				kernel.SetMemoryArgument (2, d_C);

				// Create the event wait list
				ComputeEventList eventList = new ComputeEventList ();

				// Create the command queue
				ComputeCommandQueue commands = new ComputeCommandQueue (context, context.Devices[0], ComputeCommandQueueFlags.None);

				// Execute the kernel
				commands.Execute (kernel, null, new long[] { count }, null, eventList);

				// Read back the result
				commands.ReadFromBuffer (d_C, ref h_C, false, eventList);

				// Wait until finish
				commands.Finish ();

				// Write results to file
				StreamWriter swAfter = new StreamWriter (path + "/exAfter.txt");
				for (int i = 0; i < count; ++i)
				{
					swAfter.WriteLine ("{0} + {1} -> {2}", h_A[i], h_B[i], h_C[i]);
				}

				swAfter.Flush ();
				swBefore.Flush ();
			}
		}
	}
}
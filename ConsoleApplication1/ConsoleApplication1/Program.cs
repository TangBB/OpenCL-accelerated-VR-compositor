using System;
using Cloo;
using System.IO;

namespace FlameJet
{
    class Program
    {
        public const int MAX_PARTICLES = 10;
        public const int TIME = 60;

        static void Main(string[] args)
        {
            /************************************************************************************
             *                                  OpenCL SetUp                                    *
             ************************************************************************************/
            // pick first platform
            ComputePlatform platform = ComputePlatform.Platforms[0];

            // create context with all gpu devices
            ComputeContext context = new ComputeContext(ComputeDeviceTypes.Gpu, new ComputeContextPropertyList(platform), null, IntPtr.Zero);

            // create a command queue with first gpu found
            ComputeCommandQueue queue = new ComputeCommandQueue(context, context.Devices[0], ComputeCommandQueueFlags.None);

            // load opencl source
            StreamReader streamReader = new StreamReader("../../kernels.cl");
            string clSource = streamReader.ReadToEnd();
            streamReader.Close();

            // create program with opencl source
            ComputeProgram program = new ComputeProgram(context, clSource);

            // compile opencl source
            program.Build(null, null, null, IntPtr.Zero);

            // load chosen kernel from program
            ComputeKernel kernel = program.CreateKernel("flame");

            /************************************************************************************
             *                             Setup variables and run                              *
             ************************************************************************************/
            Random rand = new Random();
            //initial position and velocity
            float px, py, pz, vx, vy, vz;
            px = py = pz = vx = vy = vz = 0f;

            //particle param
            float[] life = new float[MAX_PARTICLES];
            float[] fade = new float[MAX_PARTICLES];
            float[] x = new float[MAX_PARTICLES];
            float[] y = new float[MAX_PARTICLES];
            float[] z = new float[MAX_PARTICLES];
            float[] xi = new float[MAX_PARTICLES];
            float[] yi = new float[MAX_PARTICLES];
            float[] zi = new float[MAX_PARTICLES];
            int[] xc = new int[MAX_PARTICLES];
            int[] yc = new int[MAX_PARTICLES];
            int[] zc = new int[MAX_PARTICLES];
            float[] size = new float[MAX_PARTICLES];
            for (int i = 0; i < MAX_PARTICLES; i++)
            {
                life[i] = 1.0f;
                fade[i] = rand.Next() % 100 / 1000.0f + 0.01f;
                x[i] = px + (rand.Next() % 200 - 100) / 100.0f;
                y[i] = py + (rand.Next() % 200 - 100) / 100.0f;
                z[i] = pz + (rand.Next() % 200 - 100) / 100.0f;
                xi[i] = vx + (rand.Next() % 200 - 100) / 1000.0f;
                yi[i] = vy + (rand.Next() % 200 - 100) / 1000.0f;
                zi[i] = vz + (rand.Next() % 200 - 100) / 1000.0f;
            }

            // allocate a memory buffer with the message (the int array)
            ComputeBuffer<float> lifeBuffer = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, life);
            ComputeBuffer<float> fadeBuffer = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, fade);
            ComputeBuffer<float> xBuffer = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, x);
            ComputeBuffer<float> yBuffer = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, y);
            ComputeBuffer<float> zBuffer = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, z);
            ComputeBuffer<float> xiBuffer = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, xi);
            ComputeBuffer<float> yiBuffer = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, yi);
            ComputeBuffer<float> ziBuffer = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, zi);
            ComputeBuffer<int> xcBuffer = new ComputeBuffer<int>(context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, xc);
            ComputeBuffer<int> ycBuffer = new ComputeBuffer<int>(context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, yc);
            ComputeBuffer<int> zcBuffer = new ComputeBuffer<int>(context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, zc);
            ComputeBuffer<float> sizeBuffer = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.UseHostPointer, size);

            kernel.SetMemoryArgument(0, lifeBuffer);
            kernel.SetMemoryArgument(1, fadeBuffer);
            kernel.SetMemoryArgument(2, xBuffer);
            kernel.SetMemoryArgument(3, yBuffer);
            kernel.SetMemoryArgument(4, zBuffer);
            kernel.SetMemoryArgument(5, xiBuffer);
            kernel.SetMemoryArgument(6, yiBuffer);
            kernel.SetMemoryArgument(7, ziBuffer);
            kernel.SetMemoryArgument(8, xcBuffer);
            kernel.SetMemoryArgument(9, ycBuffer);
            kernel.SetMemoryArgument(10, zcBuffer);
            kernel.SetMemoryArgument(11, sizeBuffer);

            // execute kernel
            //dim = 1;
            long[] globalWorkOffset = { 0 };
            long[] globalWorkSize = { MAX_PARTICLES };
            long[] localWorkSize = { MAX_PARTICLES };

            for (int t = 1; t < TIME; t++)
            {
                kernel.SetValueArgument<float>(12, px);
                kernel.SetValueArgument<float>(13, py);
                kernel.SetValueArgument<float>(14, pz);
                kernel.SetValueArgument<float>(15, vx);
                kernel.SetValueArgument<float>(16, vy);
                kernel.SetValueArgument<float>(17, vz);

                queue.Execute(kernel, globalWorkOffset, globalWorkSize, localWorkSize, null);
                queue.ReadFromBuffer<float>(xBuffer, ref x, true, null);
                queue.ReadFromBuffer<float>(yBuffer, ref y, true, null);
                queue.ReadFromBuffer<float>(zBuffer, ref z, true, null);

                /*  test output */
                Console.WriteLine("t = {0}", t);
                for (int i = 0; i < MAX_PARTICLES; i++)
                {
                    Console.WriteLine("{0}, {1}, {2}", x[i], y[i], z[i]);
                }
            }

            // wait for completion
            queue.Finish();
        }
    }
}

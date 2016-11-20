__kernel void flame(__global float* life, __global float* fade,
	__global float* x, __global float* y, __global float* z, 
	__global float* xi, __global float* yi, __global float* zi, 
	__global int* xc, __global int* yc, __global int* zc,
	__global float* size,
	float px, float py, float pz,
	float vx, float vy, float vz)
{ 
	unsigned int i = get_global_id(0);
	

	life[i] -= fade[i];
	if (life[i] < 0)
	{
		 life[i] = 1.0f;
		 x[i] = px + ((int)(x[i] * 2000) % 200 - 100) / 100.0f;
		 y[i] = py + ((int)(y[i] * 2000) % 200 - 100) / 100.0f;
		 z[i] = pz + ((int)(z[i] * 2000) % 200 - 100) / 100.0f;
         xi[i] = vx + ((int)(xi[i] * 2000) % 200 - 100) / 1000.0f;
         yi[i] = vy + ((int)(yi[i] * 2000) % 200 - 100) / 1000.0f;
         zi[i] = vz + ((int)(zi[i] * 2000) % 200 - 100) / 1000.0f;
	}
	else
	{ 
		x[i] += xi[i];
		y[i] += yi[i];
		z[i] += zi[i];
		xi[i] *= 0.95f;
		yi[i] *= 0.95f;
		zi[i] *= 0.95f;
	}
	
	xc[i] = (int)(x[i] * 256) % 256;
	yc[i] = (int)(y[i] * 256) % 256;
	zc[i] = (int)(z[i] * 256) % 256;

	size[i] = 1 / (x[i] + y[i] + z[i]) / 1000;
}
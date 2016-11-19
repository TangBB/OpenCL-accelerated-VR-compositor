__kernel void particle(__global float* life, __global float* fade,
	__global float* x, __global float* y, __global float* z, 
	__global float* xi, __global float* yi, __global float* zi, 
	__global float* xg, __global float* yg, __global float* zg, 
	float px, float py, float pz,
	float vx, float vy, float vz,
	float d_ax, float d_ay, float d_az)
{ 
	unsigned int i = get_global_id(0);

	x[i] += xi[i];
	y[i] += yi[i];
	z[i] += zi[i];
	xi[i] += xg[i];
	yi[i] += yg[i];
	zi[i] += zg[i];
	xg[i] += d_ax;
	yg[i] += d_ay;
	zg[i] += d_az;

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
         xg[i] = d_ax + ((int)(xg[i] * 2000) % 200 - 100) / 10000.0f;
         yg[i] = d_ay + ((int)(yg[i] * 2000) % 200 - 100) / 10000.0f;
         zg[i] = d_az + ((int)(zg[i] * 2000) % 200 - 100) / 10000.0f;
	}
}
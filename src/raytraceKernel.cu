// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include <cutil_math.h>
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
	ray r;
	float xstep = 2 * tan( PI / 180.0f * fov.x ) / resolution.x;
	float ystep = 2 * tan( PI / 180.0f * fov.y ) / resolution.y;
		
	glm::vec3 right = -glm::cross( view, up );
	up = glm::cross( right, view );

	glm::vec3 botleft = view - (xstep*resolution.x/2)*right - (ystep*resolution.y/2)*up;

	glm::vec3 raycast = botleft + (float)x*xstep*right + (float)y*ystep*up;

	raycast = glm::normalize(raycast);

	r.origin = eye; r.direction = raycast;
	return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;      
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;     
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, 
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, 
							int* lights, int numberOfLights)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if((x<=resolution.x && y<=resolution.y)){
		ray cast = raycastFromCameraKernel( resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
		
		glm::vec3 interPoint, tmpInterPoint;
		glm::vec3 normal, tmpNormal;
		glm::vec3 dcol( 0 );
		float len = 999999.99999f, tmpLen;

		colors[index] = glm::vec3( 0 );
		int object = -1;

		for( int i = 0; i < numberOfGeoms; i ++ )
		{
			switch( geoms[i].type )
			{
			case( GEOMTYPE::CUBE ):
				tmpLen = boxIntersectionTest( geoms[i], cast, tmpInterPoint, tmpNormal );
				break;
			case( GEOMTYPE::SPHERE ):
				tmpLen = sphereIntersectionTest( geoms[i], cast, tmpInterPoint, tmpNormal );
				break;
			}
			if( tmpLen < len && tmpLen > 0 )
			{
				len = tmpLen;
				normal = tmpNormal;
				interPoint = tmpInterPoint;
				object = i;
			}
		}

		if( object == -1 )
			return;
		
		glm::vec3 light = glm::vec3( 0 );
		int matid = geoms[object].materialid;

		colors[index] = glm::vec3( 0.1 ) * materials[matid].color;

		if( materials[matid].emittance > 0 )
		{
			colors[index] = materials[matid].color;
		}
		else
		{
			ray shadowcast;
			for( int j = 0; j < numberOfLights; j ++ )
			{
				bool hasLight = true;
				//get the point to use
				glm::vec3 lightPos = multiplyMV( geoms[lights[j]].transform, glm::vec4(0,0,0,1) );
				glm::vec3 lnorm = lightPos - interPoint;


				float lDist = glm::length( lnorm );
				lnorm = glm::normalize( lnorm );

				shadowcast.direction = lnorm;
				shadowcast.origin = interPoint;

				int datmofo = -1;

				for( int i = 0; i < numberOfGeoms; i ++ )
				{
					if( i != lights[j] )
					{
						switch( geoms[i].type )
						{
						case( GEOMTYPE::CUBE ):
							tmpLen = boxIntersectionTest( geoms[i], shadowcast, tmpInterPoint, tmpNormal );
							break;
						case( GEOMTYPE::SPHERE ):
							tmpLen = sphereIntersectionTest( geoms[i], shadowcast, tmpInterPoint, tmpNormal );
							break;
						}
						if( tmpLen > 0 && tmpLen < lDist )
						{
							hasLight = false;
							datmofo = i;
							break;
						}
					}
				}
				float diffuseC = 0.1;
				if( hasLight )
				{
					diffuseC = 0.7;
				}
				float diffuse = max( (float) glm::dot( lnorm, normal ), 0.0f ) * diffuseC;
				colors[index] += materials[matid].color * materials[lights[j]].color * diffuse / (float) numberOfLights;
			}
		}
		//colors[index] = generateRandomNumberFromThread(resolution, time, x, y);
	}
}


//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  int numberOfLights = 0;
  int* lights = new int[numberOfGeoms];

  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
	if( materials[newStaticGeom.materialid].emittance > 0 )
	{
		lights[numberOfLights] = i;
		numberOfLights ++;
	}
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  material* cudamater = NULL;
  cudaMalloc((void**)&cudamater, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamater, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  int* cudalights = NULL;
  cudaMalloc((void**)&cudalights, numberOfLights*sizeof(int));
  cudaMemcpy( cudalights, lights, numberOfLights*sizeof(int), cudaMemcpyHostToDevice);

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamater, numberOfMaterials, cudalights, numberOfLights);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamater );
  cudaFree( cudalights );
  delete geomList;
  delete lights;

  // make certain the kernel has completed 
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}

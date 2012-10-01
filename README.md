-------------------------------------------------------------------------------
CIS565: Project 1: CUDA Raytracer
-------------------------------------------------------------------------------
Fall 2012
-------------------------------------------------------------------------------
Due Sunday, 09/30/2012
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
Raytracer
-------------------------------------------------------------------------------
For the ray tracer project I've implemented all of the base features for the project in an accumulation approach. Each iteration of the ray tracer renders the scene as a standard ray tracer would and then the output of this operation is accumulated and then averaged to produce one mean image at the end. I've chosen to implement in addition to the base set of features, soft shadows and depth of field. Since none of my features incorporate the bouncing of rays (no reflection, no refraction, etc.) I have implemented the ray tracer in a ray parallel format. Unfortunately, due to hardware constraints, my project runs considerably slower than I would have liked. This will be looked into more in depth in the future.

-------------------------------------------------------------------------------
Soft shadows
-------------------------------------------------------------------------------
To render my scene with soft shadows, I model the light source as being a collection of point light sources along its surface. In the code this is reflected by choosing a random point on the surface of the geometry emitting light when calculating the light ray from the intersection on the scene geometry to the light source instead of using the center of mass as a single point source. I use the iteration number as the seed for the random value and ignore the thread index. Calculating the seed this way gives the image (in my opinion) a more interesting and less grainy feel as it is being created.

-------------------------------------------------------------------------------
Depth of Field
-------------------------------------------------------------------------------
Giving the scene depth of field was a relatively straight forward task. When calculating the ray to cast into the scene I calculate it as normal, then I pivot by a small random amount about a point some distance (the focal distance, in my program it is hard coded in for simplicity to be 15 scene units). As in the soft shadows I only use the iteration number as the seed to give it a more interesting feel.

-------------------------------------------------------------------------------
Blog
-------------------------------------------------------------------------------
http://liamboone.blogspot.com/2012/09/project-1-raytracer.html

-------------------------------------------------------------------------------
Building & Running
-------------------------------------------------------------------------------
This code uses CUDA 4.2
I've tried to include all dll dependencies in the repo.
I've left the 'Release' mode in visual studio untouched to allow for ease of testing on 4.0 machines
If there are any problems please let me know, I'll be happy to do a fresh clone and run on my laptop.

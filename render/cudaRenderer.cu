#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "cycleTimer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#define TILESIZE 32
#define WARPSIZE 32
#define BLOCKSIZE 256

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

__device__ __inline__ void
shadePixelSnow(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    const float kCircleMaxAlpha = .5f;
    const float falloffScale = 4.f;

    float normPixelDist = sqrt(pixelDist) / rad;
    rgb = lookupColor(normPixelDist);

    float maxAlpha = .6f + .4f * (1.f-p.z);
    maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
    alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

__device__ __inline__ void
shadePixelNotSnow(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    // simple: each circle has an assigned color
    int index3 = 3 * circleIndex;
    rgb = *(float3*)&(cuConstRendererParams.color[index3]);
    alpha = .5f;

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // for all pixels in the bonding box
    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(index, pixelCenterNorm, p, imgPtr);
            imgPtr++;
        }
    }
}

  //////////////////
 // MY FUNCTIONS //
//////////////////
#include "circleBoxTest.cu_inl"
__global__ void
kernelFindTileCircleIntersections(int* tileCircleIntersect, int N, int s, int e) {
    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    float tileL = static_cast<float>(blockIdx.y*TILESIZE) / static_cast<float>(width);
    float tileR = fminf(1.f, static_cast<float>((blockIdx.y+1)*TILESIZE) / static_cast<float>(width));
    float tileB = static_cast<float>(blockIdx.z*TILESIZE) / static_cast<float>(height);
    float tileT = fminf(1.f, static_cast<float>((blockIdx.z+1)*TILESIZE) / static_cast<float>(height));

    int tileIndex = blockIdx.z * gridDim.y + blockIdx.y;
    int baseOffset = tileIndex * N;

    int localCircleIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int circleIndex = localCircleIndex + s;
    int circleIndex3 = 3 * circleIndex;

    if (circleIndex < e) {
        float3 p = *(float3*)(&cuConstRendererParams.position[circleIndex3]);
        float rad = cuConstRendererParams.radius[circleIndex];
        tileCircleIntersect[baseOffset + localCircleIndex] = circleInBox(p.x, p.y, rad, tileL, tileR, tileT, tileB);
    }
}

__global__ void
kernelPrintArr(int* arr, int idx, int N) {
    printf("  > [");
    for (int i = 0; i < N; i++) {
        printf("%d ", arr[idx*N+i]);
    }
    printf("]\n");
}

__global__ void
kernelPrintArrV2(int* arr, int idx, int N) {
    printf("  > [");
    for (int i = 0; i < N; i++) {
        printf("%d ", (arr+idx)[i]);
    }
    printf("]\n");
}

__device__ int
scan_warp(int* ptr, const unsigned int idx) {
    const unsigned int lane = idx % WARPSIZE;
    __syncwarp();
    for (int i = 0; i < 5; i++) {
        int shift = 1<<i;
        int tmp1 = ptr[idx];
        int tmp2 = lane >= shift ? ptr[idx - shift] : 0;
        __syncwarp();
        ptr[idx] = tmp1 + tmp2;
        __syncwarp();
    }
    return (lane > 0) ? ptr[idx-1] : 0;
}

__device__ int
scan_block(int* ptr, const unsigned int idx) {
    const unsigned int lane = idx % WARPSIZE;
    const unsigned int warp_id = idx >> 5;
    
    int baseval = ptr[idx];
    __syncthreads();

    int val = scan_warp(ptr, idx);
    __syncthreads();

    if (lane == WARPSIZE-1)
        ptr[warp_id] = ptr[idx];
    __syncthreads();

    if (warp_id == 0)
        scan_warp(ptr, idx);
    __syncthreads();

    if (warp_id > 0)
        val = val + ptr[warp_id-1];
    __syncthreads();

    ptr[idx] = val;
    __syncthreads();

    return (idx == BLOCKSIZE-1) ? val + baseval : 0;
}

__global__ void
kernelMultiExclusiveScan_SingleWarp(int* deviceArr, int length) {
    int tileIndex = blockIdx.z * gridDim.y + blockIdx.y;
    int baseOffset = tileIndex * length;

    deviceArr[baseOffset + threadIdx.x] = scan_warp(deviceArr + baseOffset, threadIdx.x);
}

void multiExclusiveScan_SingleWarp(int* deviceArr, int width, int height, int length) {
    dim3 blockDim(WARPSIZE, 1, 1);
    dim3 gridDim(1, width, height);
    kernelMultiExclusiveScan_SingleWarp<<<gridDim, blockDim>>>(deviceArr, length);
}

__global__ void
kernelMultiExclusiveScan_SingleBlock(int* deviceArr, int length) {
    int tileIndex = blockIdx.z * gridDim.y + blockIdx.y;
    int baseOffset = tileIndex * length;

    scan_block(deviceArr + baseOffset, threadIdx.x);
}

void multiExclusiveScan_SingleBlock(int* deviceArr, int width, int height, int length) {
    dim3 blockDim(BLOCKSIZE, 1, 1);
    dim3 gridDim(1, width, height);
    kernelMultiExclusiveScan_SingleBlock<<<gridDim, blockDim>>>(deviceArr, length);
}

__global__ void
kernelMultiExclusiveScan_MultiBlock(int* deviceArr, int* tempData, int tempDataLength, int length) {
    int tileIndex = blockIdx.z * gridDim.y + blockIdx.y;
    int blockInTileOffset = blockIdx.x * blockDim.x;
    int baseOffset = tileIndex * length + blockInTileOffset;
    __syncthreads();

    int val = scan_block(deviceArr + baseOffset, threadIdx.x);
    if (threadIdx.x == BLOCKSIZE-1)
        tempData[tileIndex*tempDataLength + blockIdx.x] = val;
    __syncthreads();
}

__global__ void
kernelAddTempData(int* deviceArr, int* tempData, int width, int height, int length, int tempTileLength) {
    int tileIndex = blockIdx.z * gridDim.y + blockIdx.y;
    int blockInTileOffset = blockIdx.x * blockDim.x;
    int baseOffset = tileIndex * length + blockInTileOffset;
    int tempOffset = tileIndex * tempTileLength;

    deviceArr[baseOffset + threadIdx.x] += tempData[tempOffset + blockIdx.x];
}

void multiExclusiveScan_MultiBlock(int* deviceArr, int* tempData, int width, int height, int length, int N) {
    int numBlocksPerTile = (N + BLOCKSIZE-1)/BLOCKSIZE;
    if (numBlocksPerTile <= WARPSIZE) {
        // Part 1 (WARPSIZE) - Do blocks independently
        dim3 blockDim(BLOCKSIZE, 1, 1);
        dim3 gridDim(numBlocksPerTile, width, height);
        kernelMultiExclusiveScan_MultiBlock<<<gridDim, blockDim>>>(deviceArr, tempData, WARPSIZE, length);
        cudaDeviceSynchronize();

        // Part 2 (WARPSIZE) - Add blocks together
        multiExclusiveScan_SingleWarp(tempData, width, height, WARPSIZE);
        cudaDeviceSynchronize();

        // Part 3 (WARPSIZE) - Add results back in
        blockDim = dim3(BLOCKSIZE, 1, 1);
        gridDim = dim3(numBlocksPerTile, width, height);
        kernelAddTempData<<<gridDim, blockDim>>>(deviceArr, tempData, width, height, length, WARPSIZE);
        cudaDeviceSynchronize();
    } else {
        // Part 1 (BLOCKSIZE) - Do blocks independently
        dim3 blockDim(BLOCKSIZE, 1, 1);
        dim3 gridDim(numBlocksPerTile, width, height);
        kernelMultiExclusiveScan_MultiBlock<<<gridDim, blockDim>>>(deviceArr, tempData, BLOCKSIZE, length);
        cudaDeviceSynchronize();

        // Part 2 (BLOCKSIZE) - Add blocks together
        multiExclusiveScan_SingleBlock(tempData, width, height, BLOCKSIZE);
        cudaDeviceSynchronize();

        // Part 3 (BLOCKSIZE) - Add results back in
        blockDim = dim3(BLOCKSIZE, 1, 1);
        gridDim = dim3(numBlocksPerTile, width, height);
        kernelAddTempData<<<gridDim, blockDim>>>(deviceArr, tempData, width, height, length, BLOCKSIZE);
        cudaDeviceSynchronize();
    }
}

__global__ void
kernelMultiFindStepLocs(int* steppingArr, int*  stepLocs, int* numSteps, int N, int s, int e) {
    int tileIndex = blockIdx.z * gridDim.y + blockIdx.y;
    int baseOffset = tileIndex * N;

    int localCircleIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int circleIndex = localCircleIndex + s;

    if (circleIndex < e) {
        int current = steppingArr[baseOffset + localCircleIndex];
        int next = steppingArr[baseOffset + localCircleIndex+1];
        if (next == current+1) {
            stepLocs[baseOffset + current] = circleIndex;
        }
    } else if (circleIndex == e) {
        numSteps[tileIndex] = steppingArr[baseOffset + localCircleIndex];
    }
}

__global__ void
kernelPixelUpdateSnow(int* tileCircleUpdates, int* tileNumCircles, int N) {
    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    if ((imageX >= width) || (imageY >= height))
        return;

    int pixelIdx = imageY * width + imageX;
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * pixelIdx]);
    float2 pixelCenter = make_float2(
            (0.5f + static_cast<float>(imageX)) / static_cast<float>(width),
            (0.5f + static_cast<float>(imageY)) / static_cast<float>(height));

    int tileIndex = blockIdx.y * gridDim.x + blockIdx.x;
    int baseOffset = tileIndex * N;

    int circleIndex, circleIndex3;
    float3 circlePosition;
    for (int i = 0; i < tileNumCircles[tileIndex]; i++) {
        circleIndex = tileCircleUpdates[baseOffset + i];
        circleIndex3 = 3 * circleIndex;
        circlePosition = *(float3*)(&cuConstRendererParams.position[circleIndex3]);
        shadePixelSnow(circleIndex, pixelCenter, circlePosition, imgPtr);
    }
}

__global__ void
kernelPixelUpdateNotSnow(int* tileCircleUpdates, int* tileNumCircles, int N) {
    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    if ((imageX >= width) || (imageY >= height))
        return;

    int pixelIdx = imageY * width + imageX;
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * pixelIdx]);
    float2 pixelCenter = make_float2(
            (0.5f + static_cast<float>(imageX)) / static_cast<float>(width),
            (0.5f + static_cast<float>(imageY)) / static_cast<float>(height));

    int tileIndex = blockIdx.y * gridDim.x + blockIdx.x;
    int baseOffset = tileIndex * N;

    int circleIndex, circleIndex3;
    float3 circlePosition;
    for (int i = 0; i < tileNumCircles[tileIndex]; i++) {
        circleIndex = tileCircleUpdates[baseOffset + i];
        circleIndex3 = 3 * circleIndex;
        circlePosition = *(float3*)(&cuConstRendererParams.position[circleIndex3]);
        shadePixelNotSnow(circleIndex, pixelCenter, circlePosition, imgPtr);
    }
}

////////////////////////////////////////////////////////////////////////////////////////

CudaRenderer::CudaRenderer() {
    // printf("Constructing renderer\n");
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;

    tileCircleIntersect = NULL;
    tileCircleUpdates = NULL;
    tileNumCircles = NULL;
    tempData = NULL;
}

CudaRenderer::~CudaRenderer() {
    // printf("Deconstructing renderer\n");
    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);

        cudaFree(tileCircleIntersect);
        cudaFree(tileCircleUpdates);
        cudaFree(tileNumCircles);
        cudaFree(tempData);
    }
}

const Image*
CudaRenderer::getImage() {
    // printf("Get image pointer\n");
    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * myImageWidth * myImageHeight,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    // printf("%d\n", numCircles);
    printf("Load scene\n");
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
    // Figuring out circle space allocation
    if (numCircles <= WARPSIZE-1) {
        circleSpaceAllocated = WARPSIZE;
    } else if (numCircles <= BLOCKSIZE-1) {
        circleSpaceAllocated = BLOCKSIZE;
    } else if (numCircles <= BLOCKSIZE*BLOCKSIZE-1) {
        circleSpaceAllocated = BLOCKSIZE*((numCircles + BLOCKSIZE-1)/BLOCKSIZE);
    } else {
        circleSpaceAllocated = BLOCKSIZE*BLOCKSIZE;
    }
    // printf("%d\n", numCircles);
}

void
CudaRenderer::setup() {
    // printf("Setting up\n");
    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
        printf("   Max threads per block:   %d\n", deviceProps.maxThreadsPerBlock);
        printf("   Max grid size:   (%d, %d, %d)\n", deviceProps.maxGridSize[0], deviceProps.maxGridSize[1], deviceProps.maxGridSize[2]);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * myImageWidth * myImageHeight);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Allocating new buffers
    cudaMalloc(&tileCircleIntersect, sizeof(int) * nWidthTiles * nHeightTiles * circleSpaceAllocated);
    cudaMalloc(&tileCircleUpdates, sizeof(int) * nWidthTiles * nHeightTiles * circleSpaceAllocated);
    cudaMalloc(&tileNumCircles, sizeof(int) * nWidthTiles * nHeightTiles);
    cudaMalloc(&tempData, sizeof(int) * nWidthTiles * nHeightTiles * BLOCKSIZE);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = myImageWidth;
    params.imageHeight = myImageHeight;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {
    // printf("Allocating image\n");
    if (image)
        delete image;
    image = new Image(width, height);
    myImageWidth = width;
    myImageHeight = height;
    nWidthTiles = (width + TILESIZE-1)/TILESIZE;
    nHeightTiles = (height + TILESIZE-1)/TILESIZE;
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {
    // printf("Clearing image\n");
    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (myImageWidth + blockDim.x - 1) / blockDim.x,
        (myImageHeight + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
    // printf("Advancing animation\n");
    // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

/*
void
CudaRenderer::render() {
    printf("Rendering image\n");
    // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    kernelRenderCircles<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
}
*/

void
CudaRenderer::render() {
    // printf("Rendering image %d, %d, %d\n", nWidthTiles, nHeightTiles, circleSpaceAllocated);
    dim3 blockDim;
    dim3 gridDim;
    double startTime;
    double endTime;
    // s = index of first circle rendering this iteration
    // e = index of first circle not rendering this iteration
    for (int s = 0; s < numCircles; s += BLOCKSIZE*BLOCKSIZE-1) {
        int e = (s + BLOCKSIZE*BLOCKSIZE-1 < numCircles) ? s + BLOCKSIZE*BLOCKSIZE-1 : numCircles;
        int numCirclesRendering = e - s;
        // printf("rendering %d circles (%d -> %d)\n", numCirclesRendering, s, e);

        // (1) Tile x circle intersection
        startTime = CycleTimer::currentSeconds();
        blockDim = dim3(BLOCKSIZE, 1, 1);
        gridDim = dim3((numCirclesRendering + BLOCKSIZE-1)/BLOCKSIZE, nWidthTiles, nHeightTiles);
        kernelFindTileCircleIntersections<<<gridDim, blockDim>>>(tileCircleIntersect, circleSpaceAllocated, s, e);
        cudaDeviceSynchronize();
        endTime = CycleTimer::currentSeconds();
        // printf("Step 1: %fms\n", 1000*(endTime - startTime));

        // (2) Exclusive scan
        startTime = CycleTimer::currentSeconds();
        if (numCirclesRendering <= WARPSIZE-1) {
            multiExclusiveScan_SingleWarp(tileCircleIntersect, nWidthTiles, nHeightTiles, circleSpaceAllocated);
        } else if (numCirclesRendering <= BLOCKSIZE-1) {
            multiExclusiveScan_SingleBlock(tileCircleIntersect, nWidthTiles, nHeightTiles, circleSpaceAllocated);
        } else {
            multiExclusiveScan_MultiBlock(tileCircleIntersect, tempData, nWidthTiles, nHeightTiles, circleSpaceAllocated, numCirclesRendering);
        }
        cudaDeviceSynchronize();
        endTime = CycleTimer::currentSeconds();
        // printf("Step 2: %fms\n", 1000*(endTime - startTime));

        // (3) Which circles to update
        startTime = CycleTimer::currentSeconds();
        blockDim = dim3(BLOCKSIZE, 1, 1);
        gridDim = dim3((numCirclesRendering + BLOCKSIZE-1)/BLOCKSIZE, nWidthTiles, nHeightTiles);
        kernelMultiFindStepLocs<<<gridDim, blockDim>>>(tileCircleIntersect, tileCircleUpdates, tileNumCircles, circleSpaceAllocated, s, e);
        cudaDeviceSynchronize();
        endTime = CycleTimer::currentSeconds();
        // printf("Step 3: %fms\n", 1000*(endTime - startTime));

        // (4) Update pixels
        startTime = CycleTimer::currentSeconds();
        blockDim = dim3(TILESIZE, TILESIZE);
        gridDim = dim3(nWidthTiles, nHeightTiles);
        if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
            kernelPixelUpdateSnow<<<gridDim, blockDim>>>(tileCircleUpdates, tileNumCircles, circleSpaceAllocated);
        } else {
            kernelPixelUpdateNotSnow<<<gridDim, blockDim>>>(tileCircleUpdates, tileNumCircles, circleSpaceAllocated);
        }
        cudaDeviceSynchronize();
        endTime = CycleTimer::currentSeconds();
        // printf("Step 4: %fms\n", 1000*(endTime - startTime));
    }
}
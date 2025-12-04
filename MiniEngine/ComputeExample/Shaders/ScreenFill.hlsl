// #include "ShaderUtility.hlsli"
#include "../../Core/Shaders/ShaderUtility.hlsli"
RWTexture2D<float3> ColorBuffer : register(u0);

[numthreads(8, 8, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
    uint2 pixelPos = DTid.xy;
    float2 uv = (float2)pixelPos / float2(1920.0, 1080.0); // Assuming 1080p for now, or pass constants
    
    // Simple gradient
    float3 color = float3(uv.x, uv.y, 0.5);
    
    ColorBuffer[pixelPos] = color;
}

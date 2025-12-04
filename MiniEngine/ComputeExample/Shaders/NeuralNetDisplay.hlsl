#define BLOCK_SIZE 32

cbuffer LayerParams : register(b0)
{
    uint InputChannels;
    uint OutputChannels;
    uint Width;
    uint Height;
    uint ApplyReLU;
    uint pad0;
    uint pad1;
    uint pad2;
}

StructuredBuffer<float> g_Input : register(t0);
RWTexture2D<float4> g_ScreenOutput : register(u0);

[numthreads(BLOCK_SIZE, BLOCK_SIZE, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= Width || DTid.y >= Height) return;

    uint pixelIndex = DTid.y * Width + DTid.x;
    
    // Assumes input is the final output with 3 channels
    float r = g_Input[pixelIndex * 4 + 0];
    float g = g_Input[pixelIndex * 4 + 1];
    float b = g_Input[pixelIndex * 4 + 2];
    
    g_ScreenOutput[DTid.xy] = saturate(float4(r, g, b, 1.0f));
}

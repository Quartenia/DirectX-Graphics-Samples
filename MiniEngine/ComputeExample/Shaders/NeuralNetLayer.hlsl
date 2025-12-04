#define BLOCK_SIZE 32
#define TILE_SIZE 32

cbuffer LayerParams : register(b0)
{
    uint InputChannels;
    uint OutputChannels;
    uint Width;
    uint Height;
    uint ApplyReLU; // 1 = yes, 0 = no
    uint pad0;
    uint pad1;
    uint pad2;
}

StructuredBuffer<float> g_Input : register(t0);
StructuredBuffer<float> g_Weights : register(t1);
StructuredBuffer<float> g_Biases : register(t2);
RWStructuredBuffer<float> g_Output : register(u0);

[numthreads(BLOCK_SIZE, BLOCK_SIZE, 1)]
void main(uint2 GTid : SV_GroupThreadID, uint2 GId : SV_GroupID)
{
    if (GId.x * TILE_SIZE >= Width * Height || GId.y * TILE_SIZE >= OutputChannels)
        return;

    uint2 outputIndex = uint2(GId.x * TILE_SIZE + GTid.x,
        GId.y * TILE_SIZE + GTid.y);
    
    uint bufferIndex = outputIndex.x + outputIndex.y * BLOCK_SIZE;


    float sum = 0.0f;
    for (uint inCh = 0; inCh < InputChannels; ++inCh)
    {
        float inVal = g_Input[outputIndex.x * InputChannels + inCh];
            // Weights matrix is [InputChannels x OutputChannels] flattened
            // Row-major: W[inCh][outCh]
        float weight = g_Weights[inCh * OutputChannels + outputIndex.y];
        sum += inVal * weight;
    }
    sum += g_Biases[outputIndex.y];

    if (ApplyReLU)
        sum = max(0.0f, sum);

    g_Output[outputIndex.x * OutputChannels + outputIndex.y] = sum;
  
}

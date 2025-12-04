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

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void main(uint2 groupThreadId : SV_GroupThreadID, uint2 groupId : SV_GroupID)
{
    if (groupId.x * TILE_SIZE + groupThreadId.x >= Width * Height || groupId.y * TILE_SIZE + groupThreadId.y >= OutputChannels)
        return;

    uint2 outputIndex = uint2(groupId.x * TILE_SIZE + groupThreadId.x,
        groupId.y * TILE_SIZE + groupThreadId.y);
    
    uint batchSize = Width * Height;
    

    float sum = 0.0f;
    for (uint inCh = 0; inCh < InputChannels; ++inCh)
    {
        // input: batchSize x inputChannels
        // weight: inputChannels x outputChannels
        float inVal = g_Input[outputIndex.x * InputChannels + inCh];
        // Row-major: W[inCh][outCh]
        float weight = g_Weights[inCh * OutputChannels + outputIndex.y];
        //float weight = g_Weights[outputIndex.y * InputChannels + inCh];
        sum += inVal * weight;
    }
    sum += g_Biases[outputIndex.y];

    if (ApplyReLU)
        sum = max(0.0f, sum);

    g_Output[outputIndex.x * OutputChannels + outputIndex.y] = sum;
  
}

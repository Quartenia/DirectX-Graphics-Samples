static const uint kTileSize = 32;

cbuffer LayerParams : register(b0)
{
    uint M;
    uint K;
    uint N;
    uint ApplyReLU;
}

// M x K
StructuredBuffer<float> g_Input : register(t0); 
// K x N
StructuredBuffer<float> g_Weights : register(t1); 
// N
StructuredBuffer<float> g_Biases : register(t2); 
// M x N
RWStructuredBuffer<float> g_Output : register(u0); 

groupshared float gs_Input[kTileSize][kTileSize];
groupshared float gs_Weights[kTileSize][kTileSize];

uint CeilDiv(uint numerator, uint denominator)
{
    return (numerator + denominator - 1) / denominator;
}

[numthreads(kTileSize, kTileSize, 1)]
void main(uint2 groupThreadId : SV_GroupThreadID, uint2 groupId : SV_GroupID)
{
    // TODO: flat dispatch and coalesced writes
    uint m = groupId.x * kTileSize + groupThreadId.y;
    uint n = groupId.y * kTileSize + groupThreadId.x;
    
    float y = 0.0f;
    uint kTiles = CeilDiv(K, kTileSize);
    
    // zero the gs
    gs_Weights[groupThreadId.y][groupThreadId.x] = 0.0;
    GroupMemoryBarrierWithGroupSync();
    
    
    for (uint tK = 0; tK < kTiles; ++tK)
    {
        uint kI = tK * kTileSize + groupThreadId.x;
        if (kI < K && m < M)
        {
            gs_Input[groupThreadId.x][groupThreadId.y] = g_Input[m * K + kI];
        }
        else
        {
            gs_Input[groupThreadId.x][groupThreadId.y] = 0.0;
        }
        
        uint kW = tK * kTileSize + groupThreadId.y;
        if (kW < K && n < N)
        {
            gs_Weights[groupThreadId.y][groupThreadId.x] = g_Weights[kW * N + n];
        }
        else
        {
            gs_Weights[groupThreadId.y][groupThreadId.x] = 0.0;

        }
        
        GroupMemoryBarrierWithGroupSync();

        for (uint k = 0; k < kTileSize; ++k)
        {   
            float x = gs_Input[k][groupThreadId.y];
            float w = gs_Weights[k][groupThreadId.x];
            y += x * w;
        }
        GroupMemoryBarrierWithGroupSync();

    }
   

    if (n < N && m < M)
    {
        y += g_Biases[n];

        if (ApplyReLU)
            y = max(0.0f, y);
        g_Output[m * N + n] = y;
        
    }
    
}

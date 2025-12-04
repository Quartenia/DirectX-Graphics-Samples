#define TILE_SIZE 32

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

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void main(uint2 groupThreadId : SV_GroupThreadID, uint2 groupId : SV_GroupID)
{
    uint m = groupId.x * TILE_SIZE + groupThreadId.x;
    uint n = groupId.y * TILE_SIZE + groupThreadId.y;
        
    if (m >= M || n >= N)
        return;
    
    float y = 0.0f;
    for (uint k = 0; k < K; ++k)
    {
        float x = g_Input[m * K + k];
        float w = g_Weights[k * N + n];
        y += x * w;
    }
    y += g_Biases[n];

    if (ApplyReLU)
        y = max(0.0f, y);

    g_Output[m * N + n] = y;
}

#include "pch.h"
#include "GameCore.h"
#include "GraphicsCore.h"
#include "SystemTime.h"
#include "TextRenderer.h"
#include "GameInput.h"
#include "CommandContext.h"
#include "RootSignature.h"
#include "PipelineState.h"
#include "BufferManager.h"
#include "GpuBuffer.h"
#include "Display.h"
#include "PostEffects.h"
#include "FXAA.h"
#include "DepthOfField.h"
#include "ModelWeights.h"

#include "CompiledShaders/NeuralNetLayer.h"
#include "CompiledShaders/NeuralNetDisplay.h"

using namespace GameCore;
using namespace Graphics;


class ComputeExample : public GameCore::IGameApp
{
public:

    ComputeExample()
    {
        // Set startup resolution for both display and rendering
        g_DisplayWidth = 1280;
        g_DisplayHeight = 720;
    }

    virtual void Startup( void ) override;
    virtual void Cleanup( void ) override;

    virtual void Update( float deltaT ) override;
    virtual void RenderScene( void ) override;

private:
    RootSignature m_RootSig;
    ComputePSO m_LayerPSO;
    ComputePSO m_DisplayPSO;

    // Buffers
    StructuredBuffer m_InputUV;
    
    // Layer 1 (2 -> 32)
    StructuredBuffer m_Weights_L1;
    StructuredBuffer m_Biases_L1;
    StructuredBuffer m_Intermediate_1;

    // Layer 2 (32 -> 32)
    StructuredBuffer m_Weights_L2;
    StructuredBuffer m_Biases_L2;
    StructuredBuffer m_Intermediate_2;

    // Layer 3 (32 -> 32)
    StructuredBuffer m_Weights_L3;
    StructuredBuffer m_Biases_L3;
    StructuredBuffer m_Intermediate_3;

    // Layer 4 (32 -> 3)
    StructuredBuffer m_Weights_L4;
    StructuredBuffer m_Biases_L4;
    StructuredBuffer m_FinalOutput;
};

CREATE_APPLICATION( ComputeExample )

struct alignas(16) LayerParams
{
    uint32_t InputChannels;
    uint32_t OutputChannels;
    uint32_t Width;
    uint32_t Height;

    uint32_t ApplyReLU;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

void ComputeExample::Startup( void )
{
    // Initialize rendering buffers to match display resolution
    InitializeRenderingBuffers(g_DisplayWidth, g_DisplayHeight);

    // Disable post-processing effects
    PostEffects::EnableHDR = false;
    PostEffects::BloomEnable = false;
    PostEffects::EnableAdaptation = false;
    FXAA::Enable = false;
    DepthOfField::Enable = false;

    m_RootSig.Reset(4, 0);
    m_RootSig[0].InitAsConstantBuffer(0); // LayerParams
    m_RootSig[1].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1); // Input
    m_RootSig[2].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1); // Weights
    m_RootSig[3].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 1); // Biases
    // UAV is usually dynamic descriptor in MiniEngine or root descriptor?
    // MiniEngine ComputeContext::SetDynamicDescriptor takes (rootIndex, offset, handle).
    // But we need a UAV for output.
    // Let's add a UAV range.
    // Wait, MiniEngine usually binds UAVs via SetDynamicDescriptor if the root param allows it.
    // Let's check how ScreenFill did it.
    // m_RootSig[0].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 1);
    // So we need a UAV slot.
    // Let's add it as slot 4.
    // Actually, let's reorganize:
    // 0: CBV (b0)
    // 1: SRV Input (t0)
    // 2: SRV Weights (t1)
    // 3: SRV Biases (t2)
    // 4: UAV Output (u0)
    
    // Re-init root sig
    m_RootSig.Reset(5, 0);
    m_RootSig[0].InitAsConstantBuffer(0);
    m_RootSig[1].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1);
    m_RootSig[2].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);
    m_RootSig[3].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 1);
    m_RootSig[4].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 1);
    m_RootSig.Finalize(L"NeuralNet");

    m_LayerPSO.SetRootSignature(m_RootSig);
    m_LayerPSO.SetComputeShader(g_pNeuralNetLayer, sizeof(g_pNeuralNetLayer));
    m_LayerPSO.Finalize();

    m_DisplayPSO.SetRootSignature(m_RootSig); // Reusing same root sig structure, though some slots unused
    m_DisplayPSO.SetComputeShader(g_pNeuralNetDisplay, sizeof(g_pNeuralNetDisplay));
    m_DisplayPSO.Finalize();

    // Initialize Buffers
    uint32_t width = g_SceneColorBuffer.GetWidth();
    uint32_t height = g_SceneColorBuffer.GetHeight();
    uint32_t pixelCount = width * height;

    // Input UV
    std::vector<float> inputUV(pixelCount * 2);
    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            inputUV[(y * width + x) * 2 + 0] = (float)x / width * 2.0 - 1.0;
            inputUV[(y * width + x) * 2 + 1] = (float)y / height * 2.0 - 1.0;
        }
    }
    m_InputUV.Create(L"Input UV", pixelCount * 2, sizeof(float), inputUV.data());

    // Weights & Biases Helper
    auto CreateWeights = [&](StructuredBuffer& buf, const std::vector<float>& data, const std::wstring& name, uint32_t inCh, uint32_t outCh) {
        buf.Create(name, inCh * outCh, sizeof(float), data.data());
    };
    auto CreateBiases = [&](StructuredBuffer& buf, const std::vector<float>& data, const std::wstring& name, uint32_t count) {
        buf.Create(name, count, sizeof(float), data.data());
    };
    auto CreateInter = [&](StructuredBuffer& buf, const std::wstring& name, uint32_t channels) {
        buf.Create(name, pixelCount * channels, sizeof(float)); // No init data
    };

    // Layer 1
    CreateWeights(m_Weights_L1, ModelWeights::net_0_weight, L"Weights L1", 2, 32);
    CreateBiases(m_Biases_L1, ModelWeights::net_0_bias, L"Biases L1", 32);
    CreateInter(m_Intermediate_1, L"Inter L1", 32);

    // Layer 2
    CreateWeights(m_Weights_L2, ModelWeights::net_2_weight, L"Weights L2", 32, 32);
    CreateBiases(m_Biases_L2, ModelWeights::net_2_weight, L"Biases L2", 32);
    CreateInter(m_Intermediate_2, L"Inter L2", 32);

    // Layer 3
    CreateWeights(m_Weights_L3, ModelWeights::net_4_weight, L"Weights L3", 32, 32);
    CreateBiases(m_Biases_L3, ModelWeights::net_4_weight, L"Biases L3", 32);
    CreateInter(m_Intermediate_3, L"Inter L3", 32);

    // Layer 4
    CreateWeights(m_Weights_L4, ModelWeights::net_6_weight, L"Weights L4", 32, 4);
    CreateBiases(m_Biases_L4, ModelWeights::net_6_weight, L"Biases L4", 4);
    CreateInter(m_FinalOutput, L"Final Output", 4);
}

void ComputeExample::Cleanup( void )
{
    m_InputUV.Destroy();
    m_Weights_L1.Destroy(); m_Biases_L1.Destroy(); m_Intermediate_1.Destroy();
    m_Weights_L2.Destroy(); m_Biases_L2.Destroy(); m_Intermediate_2.Destroy();
    m_Weights_L3.Destroy(); m_Biases_L3.Destroy(); m_Intermediate_3.Destroy();
    m_Weights_L4.Destroy(); m_Biases_L4.Destroy(); m_FinalOutput.Destroy();
}

void ComputeExample::Update( float /*deltaT*/ )
{
}

void ComputeExample::RenderScene( void )
{
    ComputeContext& Context = ComputeContext::Begin(L"Neural Net Render");

    Context.SetRootSignature(m_RootSig);
    Context.SetPipelineState(m_LayerPSO);

    uint32_t width = g_SceneColorBuffer.GetWidth();
    uint32_t height = g_SceneColorBuffer.GetHeight();

    auto DispatchLayer = [&](StructuredBuffer& input, StructuredBuffer& weights, StructuredBuffer& biases, StructuredBuffer& output, uint32_t inCh, uint32_t outCh, bool relu)
    {
        LayerParams params = { inCh, outCh, width, height, (uint32_t)relu };
        Context.SetDynamicConstantBufferView(0, sizeof(params), &params);
        
        Context.TransitionResource(input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Context.TransitionResource(weights, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Context.TransitionResource(biases, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Context.TransitionResource(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        Context.SetDynamicDescriptor(1, 0, input.GetSRV());
        Context.SetDynamicDescriptor(2, 0, weights.GetSRV());
        Context.SetDynamicDescriptor(3, 0, biases.GetSRV());
        Context.SetDynamicDescriptor(4, 0, output.GetUAV());

        Context.Dispatch2D(width*height, outCh, 32, 32);
    };

    // L1
    DispatchLayer(m_InputUV, m_Weights_L1, m_Biases_L1, m_Intermediate_1, 2, 32, true);
    Context.InsertUAVBarrier(m_Intermediate_1);

    // L2
    DispatchLayer(m_Intermediate_1, m_Weights_L2, m_Biases_L2, m_Intermediate_2, 32, 32, true);
    Context.InsertUAVBarrier(m_Intermediate_2);

    // L3
    DispatchLayer(m_Intermediate_2, m_Weights_L3, m_Biases_L3, m_Intermediate_3, 32, 32, true);
    Context.InsertUAVBarrier(m_Intermediate_3);

    // L4
    DispatchLayer(m_Intermediate_3, m_Weights_L4, m_Biases_L4, m_FinalOutput, 32, 4, false);
    Context.InsertUAVBarrier(m_FinalOutput);

    // Display
    Context.SetPipelineState(m_DisplayPSO);
    LayerParams displayParams = { 4, 0, width, height, 0 }; // Only width/height matter
    Context.SetDynamicConstantBufferView(0, sizeof(displayParams), &displayParams);

    Context.TransitionResource(m_FinalOutput, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Context.TransitionResource(g_SceneColorBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    Context.SetDynamicDescriptor(1, 0, m_FinalOutput.GetSRV());
    Context.SetDynamicDescriptor(4, 0, g_SceneColorBuffer.GetUAV()); // Using slot 4 (u0) for screen output

    Context.Dispatch2D(width, height, 32, 32);

    Context.Finish();
}

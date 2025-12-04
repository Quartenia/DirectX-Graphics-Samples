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

#include "CompiledShaders/ScreenFill.h"

using namespace GameCore;
using namespace Graphics;

class ComputeExample : public GameCore::IGameApp
{
public:

    ComputeExample()
    {
    }

    virtual void Startup( void ) override;
    virtual void Cleanup( void ) override;

    virtual void Update( float deltaT ) override;
    virtual void RenderScene( void ) override;

private:
    RootSignature m_RootSig;
    ComputePSO m_ComputePSO;
};

CREATE_APPLICATION( ComputeExample )

void ComputeExample::Startup( void )
{
    m_RootSig.Reset(1, 0);
    m_RootSig[0].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 1);
    m_RootSig.Finalize(L"Compute Example");

    m_ComputePSO.SetRootSignature(m_RootSig);
    m_ComputePSO.SetComputeShader(g_pScreenFill, sizeof(g_pScreenFill));
    m_ComputePSO.Finalize();
}

void ComputeExample::Cleanup( void )
{
}

void ComputeExample::Update( float /*deltaT*/ )
{
}

void ComputeExample::RenderScene( void )
{
    ComputeContext& Context = ComputeContext::Begin(L"Scene Render");

    Context.SetRootSignature(m_RootSig);
    Context.SetPipelineState(m_ComputePSO);

    Context.TransitionResource(g_SceneColorBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Context.SetDynamicDescriptor(0, 0, g_SceneColorBuffer.GetUAV());

    Context.Dispatch2D(g_SceneColorBuffer.GetWidth(), g_SceneColorBuffer.GetHeight());

    Context.Finish();
}

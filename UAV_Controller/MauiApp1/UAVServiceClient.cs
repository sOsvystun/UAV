using System;
using System.Threading.Tasks;
using UAV_MAUI_DJI_Client;

namespace UAV_MAUI_DJI_Client.Services;

public class UAVServiceClient
{
    private readonly UAVService.UAVServiceClient _client;

    public UAVServiceClient(string serverAddress)
    {
        var channel = GrpcChannel.ForAddress(serverAddress);
        _client = new UAVService.UAVServiceClient(channel);
    }

    public async Task<TrajectoryResponse> GetTrajectoryAsync(TrajectoryRequest request)
    {
        return await _client.GetTrajectoryAsync(request);
    }

    public async Task<DefectDetectionResponse> DetectDefectsAsync(DefectDetectionRequest request)
    {
        return await _client.DetectDefectsAsync(request);
    }

    public async Task<CriticalityAssessmentResponse> AssessCriticalityAsync(CriticalityAssessmentRequest request)
    {
        return await _client.AssessCriticalityAsync(request);
    }

    public async Task<ReportResponse> GenerateReportAsync(ReportRequest request)
    {
        return await _client.GenerateReportAsync(request);
    }
}
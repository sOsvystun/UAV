namespace UAV_MAUI_DJI_Client.Services;

public class UAVInspectionManager
{
    private readonly UAVServiceClient _serviceClient;

    public UAVInspectionManager(string serverAddress)
    {
        _serviceClient = new UAVServiceClient(serverAddress);
    }

    public async Task PerformInspectionAsync(TrajectoryRequest trajectoryRequest, Image capturedImage, string turbineId)
    {
        var trajectoryResponse = await _serviceClient.GetTrajectoryAsync(trajectoryRequest);

        var detectionRequest = new DefectDetectionRequest { Image = capturedImage };
        var detectionResponse = await _serviceClient.DetectDefectsAsync(detectionRequest);

        var criticalityRequest = new CriticalityAssessmentRequest();
        foreach (var defect in detectionResponse.Defects)
        {
            criticalityRequest.Defects.Add(new DefectInfo { Type = defect.Type, Confidence = defect.Confidence });
        }

        var criticalityResponse = await _serviceClient.AssessCriticalityAsync(criticalityRequest);

        var reportRequest = new ReportRequest
        {
            TurbineId = turbineId,
            Trajectory = trajectoryResponse,
            Defects = detectionResponse,
            Criticalities = criticalityResponse
        };

        var reportResponse = await _serviceClient.GenerateReportAsync(reportRequest);

        Console.WriteLine($"Inspection report generated: {reportResponse.ReportUrl}");
    }
}
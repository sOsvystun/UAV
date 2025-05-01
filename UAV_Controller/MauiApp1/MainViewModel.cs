using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Grpc.Net.Client;
using System;
using System.Threading.Tasks;
using UAV_MAUI_DJI_Client.Trajectory;
using UAV_MAUI_DJI_Client.Detection;
using UAV_MAUI_DJI_Client.Criticality;
using UAV_MAUI_DJI_Client.Reporting;
using DJI.SDK;
using System.Collections.Generic;

namespace UAV_MAUI_DJI_Client
{
    public partial class MainViewModel : ObservableObject
    {
        private readonly TrajectoryPlanner.TrajectoryPlannerClient _trajectoryClient;
        private readonly DefectDetection.DefectDetectionClient _detectionClient;
        private readonly CriticalityAssessment.CriticalityAssessmentClient _criticalityClient;
        private readonly ReportingService.ReportingServiceClient _reportingClient;
        private readonly CameraController _cameraController;
        private DJIController _droneController;

        public MainViewModel()
        {
            var channel = GrpcChannel.ForAddress("https://192.168.0.1:5051", new GrpcChannelOptions);
            {
                HttpHandler = new HttpClientHandler
                {
                    ServerCertificateCustomValidationCallback = (message, cert, chain, errors) => true
                }
            });
            _trajectoryClient = new TrajectoryPlanner.TrajectoryPlannerClient(channel);
            _detectionClient = new DefectDetection.DefectDetectionClient(channel);
            _criticalityClient = new CriticalityAssessment.CriticalityAssessmentClient(channel);
            _reportingClient = new ReportingService.ReportingServiceClient(channel);
            _cameraController = new CameraController();
            _droneController = new DJIController();
        }

        [RelayCommand]
        public async Task StartInspectionAsync()
        {
            try
            {
                await _droneController.ConnectAsync();

                var trajectoryRequest = new TrajectoryRequest
                {
                    TurbineGeometry = new TurbineGeometry { /* populate data */ },
                    WindSpeed = 12.5,
                    WindDirection = 90
                };

                var trajectoryResponse = await _trajectoryClient.GetInspectionTrajectoryAsync(trajectoryRequest);

                List<DefectDetectionResponse> allDetectionResponses = new();
                List<CriticalityAssessmentResponse> allCriticalityResponses = new();

                foreach (var waypoint in trajectoryResponse.Waypoints)
                {
                    await _droneController.FlyToAsync(new DJILocation
                    {
                        Latitude = waypoint.X,
                        Longitude = waypoint.Y,
                        Altitude = waypoint.Z
                    });

                    var rgbImage = await _cameraController.CaptureRgbImageAsync(_droneController);
                    var thermalImage = await _cameraController.CaptureThermalImageAsync(_droneController);
                    var combinedImage = await _cameraController.CombineImagesAsync(rgbImage, thermalImage);

                    var detectionResponse = await DetectDefectsAsync(combinedImage);
                    allDetectionResponses.Add(detectionResponse);

                    var criticalityResponse = await AssessCriticalityAsync(detectionResponse);
                    allCriticalityResponses.Add(criticalityResponse);
                }

                await GenerateAndSaveReportAsync(trajectoryResponse, allDetectionResponses, allCriticalityResponses);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error during inspection: {ex.Message}");
            }
            finally
            {
                await _droneController.DisconnectAsync();
            }
        }

        private async Task<DefectDetectionResponse> DetectDefectsAsync(byte[] imageData)
        {
            var detectionRequest = new DefectDetectionRequest
            {
                Image = new Image { Data = Google.Protobuf.ByteString.CopyFrom(imageData) }
            };

            return await _detectionClient.DetectDefectsAsync(detectionRequest);
        }

        private async Task<CriticalityAssessmentResponse> AssessCriticalityAsync(DefectDetectionResponse detectionResponse)
        {
            var criticalityRequest = new CriticalityAssessmentRequest();

            foreach (var defect in detectionResponse.Defects)
            {
                criticalityRequest.Defects.Add(new DefectInfo
                {
                    Type = defect.Type,
                    Confidence = defect.Confidence
                });
            }

            return await _criticalityClient.AssessCriticalityAsync(criticalityRequest);
        }

        private async Task GenerateAndSaveReportAsync(TrajectoryResponse trajectoryResponse, List<DefectDetectionResponse> detectionResponses, List<CriticalityAssessmentResponse> criticalityResponses)
        {
            var reportRequest = new ReportRequest
            {
                TurbineId = "Turbine-123",
                Trajectory = trajectoryResponse
            };

            foreach (var detectionResponse in detectionResponses)
            {
                reportRequest.Defects.MergeFrom(detectionResponse);
            }

            foreach (var criticalityResponse in criticalityResponses)
            {
                reportRequest.Criticalities.MergeFrom(criticalityResponse);
            }

            var reportResponse = await _reportingClient.GenerateReportAsync(reportRequest);

            Console.WriteLine($"Report generated: {reportResponse.ReportUrl}");
        }
    }

    public class CameraController
    {
        public async Task<byte[]> CaptureRgbImageAsync(DJIController droneController)
        {
            return await droneController.CaptureImageAsync(CameraType.RGB);
        }

        public async Task<byte[]> CaptureThermalImageAsync(DJIController droneController)
        {
            return await droneController.CaptureImageAsync(CameraType.Thermal);
        }

        public async Task<byte[]> CombineImagesAsync(byte[] rgbImage, byte[] thermalImage)
        {
            await Task.Delay(100);
            byte[] combinedImage = new byte[Math.Min(rgbImage.Length, thermalImage.Length)];
            for (int i = 0; i < combinedImage.Length; i++)
            {
                combinedImage[i] = (byte)((rgbImage[i] * 0.6) + (thermalImage[i] * 0.4));
            }
            return combinedImage;
        }
    }

    public enum CameraType
    {
        RGB,
        Thermal
    }
}

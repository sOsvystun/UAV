using System;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace UAV_MAUI_DJI_Client.Services
{
    public class DJIFlyControllerManager
    {
        private DJIController _djiController;

        public DJIFlyControllerManager()
        {
            _djiController = new DJIController();
        }

        public async Task ConnectAsync()
        {
            if (!_djiController.IsConnected)
            {
                await _djiController.ConnectAsync();
                Console.WriteLine("DJI Drone Connected.");
            }
        }

        public async Task DisconnectAsync()
        {
            if (_djiController.IsConnected)
            {
                await _djiController.DisconnectAsync();
                Console.WriteLine("DJI Drone Disconnected.");
            }
        }

        public async Task FlyToWaypointAsync(DJILocation waypoint)
        {
            await _djiController.FlyToAsync(waypoint);
            Console.WriteLine($"Flying to Waypoint: Lat {waypoint.Latitude}, Lon {waypoint.Longitude}, Alt {waypoint.Altitude}");
        }

        public async Task<List<byte[]>> CaptureMultipleImagesAsync(CameraType cameraType, int imageCount, int intervalSeconds)
        {
            List<byte[]> images = new();
            for (int i = 0; i < imageCount; i++)
            {
                var image = await _djiController.CaptureImageAsync(cameraType);
                images.Add(image);
                Console.WriteLine($"Captured {cameraType} image {i + 1}/{imageCount}");
                await Task.Delay(intervalSeconds * 1000);
            }
            return images;
        }

        public async Task<byte[]> CaptureSingleImageAsync(CameraType cameraType)
        {
            var image = await _djiController.CaptureImageAsync(cameraType);
            Console.WriteLine($"Captured single {cameraType} image.");
            return image;
        }

        public async Task<DJITelemetry> GetTelemetryAsync()
        {
            var telemetry = await _djiController.GetTelemetryAsync();
            Console.WriteLine($"Telemetry: Altitude {telemetry.Altitude}m, Speed {telemetry.Speed}m/s, Battery {telemetry.BatteryLevel}%");
            return telemetry;
        }

        public async Task ExecuteFlightPlanAsync(List<DJILocation> waypoints, CameraType cameraType, int imagesPerWaypoint, int intervalSeconds)
        {
            foreach (var waypoint in waypoints)
            {
                await FlyToWaypointAsync(waypoint);
                await CaptureMultipleImagesAsync(cameraType, imagesPerWaypoint, intervalSeconds);
            }
        }

        public async Task EmergencyReturnToHomeAsync()
        {
            await _djiController.ReturnToHomeAsync();
            Console.WriteLine("Drone returning to home.");
        }

        public async Task StartVideoRecordingAsync(CameraType cameraType)
        {
            await _djiController.StartVideoRecordingAsync(cameraType);
            Console.WriteLine($"Started video recording with {cameraType} camera.");
        }

        public async Task StopVideoRecordingAsync(CameraType cameraType)
        {
            await _djiController.StopVideoRecordingAsync(cameraType);
            Console.WriteLine($"Stopped video recording with {cameraType} camera.");
        }

        public async Task RotateDroneAsync(double yawAngle)
        {
            await _djiController.RotateAsync(yawAngle);
            Console.WriteLine($"Drone rotated by {yawAngle} degrees.");
        }

        public async Task HoverAsync(int durationSeconds)
        {
            await _djiController.HoverAsync(durationSeconds);
            Console.WriteLine($"Drone hovering for {durationSeconds} seconds.");
        }

        public async Task SetGimbalAngleAsync(double pitch, double roll, double yaw)
        {
            await _djiController.SetGimbalAngleAsync(pitch, roll, yaw);
            Console.WriteLine($"Gimbal angles set to Pitch: {pitch}, Roll: {roll}, Yaw: {yaw}.");
        }

        public async Task<List<DJITelemetry>> GetTelemetryLogAsync(int intervalSeconds, int durationSeconds)
        {
            List<DJITelemetry> telemetryLog = new();
            int count = durationSeconds / intervalSeconds;
            for (int i = 0; i < count; i++)
            {
                telemetryLog.Add(await GetTelemetryAsync());
                await Task.Delay(intervalSeconds * 1000);
            }
            Console.WriteLine("Telemetry log completed.");
            return telemetryLog;
        }
    }
}

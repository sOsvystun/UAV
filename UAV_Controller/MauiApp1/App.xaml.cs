namespace MauiApp1;

public partial class App : Application
{
    public App()
    {
        InitializeComponent();

        MainPage = new AppShell();
    }

    protected override Window CreateWindow(IActivationState? activationState)
    {
        var window = base.CreateWindow(activationState);

        window.Created += async (sender, args) =>
        {
            await InitializeServicesAsync();
        };

        return window;
    }

    private async Task InitializeServicesAsync()
    {
        try
        {
            // Initialize DJI Controller
            var djiControllerManager = new Services.DJIFlyControllerManager();
            await djiControllerManager.ConnectAsync();

            // Initialize gRPC Service Client
            var uavServiceClient = new Services.UAVServiceClient("https://your-rust-server-address");

            // Preload expert parameters
            var expertParams = Services.DefectExpertParams.Load("expert_data.json");

            Console.WriteLine("All services initialized successfully.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Initialization error: {ex.Message}");
        }
    }
}
/**
 * Secure Subprocess Manager
 * ========================
 * 
 * This header defines a secure subprocess execution system to replace
 * dangerous system() calls that are vulnerable to command injection.
 * 
 * Key Features:
 * - Command validation and sanitization
 * - Timeout and resource limit controls
 * - Secure environment variable handling
 * - Output capture and logging
 * - Process isolation and sandboxing
 * 
 * Author: Security Framework Team
 */

#ifndef SECURE_SUBPROCESS_MANAGER_H
#define SECURE_SUBPROCESS_MANAGER_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>

#ifdef _WIN32
    #include <windows.h>
    #include <process.h>
#else
    #include <unistd.h>
    #include <sys/wait.h>
    #include <signal.h>
    #include <fcntl.h>
#endif

namespace security {

/**
 * Process execution result containing output and metadata
 */
struct ProcessResult {
    int exit_code;
    std::string stdout_output;
    std::string stderr_output;
    std::chrono::milliseconds execution_time;
    bool timed_out;
    bool success;
    std::string error_message;
    
    ProcessResult() : exit_code(-1), execution_time(0), timed_out(false), success(false) {}
};

/**
 * Configuration for subprocess execution
 */
struct ProcessConfig {
    std::vector<std::string> command;                    // Command and arguments
    std::map<std::string, std::string> environment;     // Environment variables
    std::string working_directory;                       // Working directory
    std::chrono::milliseconds timeout;                   // Execution timeout
    bool capture_output;                                 // Capture stdout/stderr
    bool inherit_environment;                            // Inherit parent environment
    size_t max_output_size;                             // Maximum output size (bytes)
    std::vector<std::string> allowed_commands;          // Whitelist of allowed commands
    
    ProcessConfig() 
        : timeout(std::chrono::milliseconds(30000))
        , capture_output(true)
        , inherit_environment(false)
        , max_output_size(1024 * 1024)  // 1MB default
    {}
};

/**
 * Security policy for subprocess execution
 */
struct SecurityPolicy {
    std::vector<std::string> command_whitelist;         // Allowed commands
    std::vector<std::string> command_blacklist;         // Forbidden commands
    std::vector<std::string> allowed_paths;             // Allowed execution paths
    std::vector<std::string> forbidden_patterns;        // Forbidden argument patterns
    bool require_absolute_paths;                         // Require absolute command paths
    bool validate_command_signatures;                    // Validate executable signatures
    size_t max_arguments;                               // Maximum number of arguments
    size_t max_argument_length;                         // Maximum argument length
    
    SecurityPolicy()
        : require_absolute_paths(true)
        , validate_command_signatures(false)
        , max_arguments(100)
        , max_argument_length(1024)
    {}
};

/**
 * Secure subprocess manager class
 */
class SecureSubprocessManager {
public:
    /**
     * Constructor with security policy
     */
    explicit SecureSubprocessManager(const SecurityPolicy& policy = SecurityPolicy());
    
    /**
     * Destructor - cleanup any running processes
     */
    ~SecureSubprocessManager();
    
    /**
     * Execute command securely with full validation
     * 
     * @param config Process configuration
     * @return ProcessResult with execution details
     */
    ProcessResult executeSecurely(const ProcessConfig& config);
    
    /**
     * Execute simple command with default configuration
     * 
     * @param command Command to execute
     * @param args Command arguments
     * @param timeout_ms Timeout in milliseconds
     * @return ProcessResult with execution details
     */
    ProcessResult executeCommand(const std::string& command, 
                               const std::vector<std::string>& args = {},
                               int timeout_ms = 30000);
    
    /**
     * Validate command against security policy
     * 
     * @param command Full command path
     * @param args Command arguments
     * @return true if command is allowed, false otherwise
     */
    bool validateCommand(const std::string& command, const std::vector<std::string>& args);
    
    /**
     * Sanitize file path to prevent directory traversal
     * 
     * @param path File path to sanitize
     * @param base_directory Base directory to restrict to
     * @return Sanitized path or empty string if invalid
     */
    std::string sanitizePath(const std::string& path, const std::string& base_directory = "");
    
    /**
     * Sanitize command arguments
     * 
     * @param args Arguments to sanitize
     * @return Sanitized arguments
     */
    std::vector<std::string> sanitizeArguments(const std::vector<std::string>& args);
    
    /**
     * Kill all running processes managed by this instance
     */
    void killAllProcesses();
    
    /**
     * Get number of currently running processes
     */
    size_t getRunningProcessCount() const;
    
    /**
     * Update security policy
     */
    void updateSecurityPolicy(const SecurityPolicy& policy);
    
    /**
     * Enable/disable debug logging
     */
    void setDebugMode(bool enabled);

private:
    SecurityPolicy security_policy_;
    mutable std::mutex processes_mutex_;
    std::vector<std::thread> process_threads_;
    std::atomic<bool> debug_mode_;
    
    // Platform-specific process handles
#ifdef _WIN32
    std::vector<HANDLE> process_handles_;
#else
    std::vector<pid_t> process_pids_;
#endif
    
    /**
     * Validate command path and executable
     */
    bool validateCommandPath(const std::string& command);
    
    /**
     * Validate command arguments against security policy
     */
    bool validateArguments(const std::vector<std::string>& args);
    
    /**
     * Check for dangerous patterns in arguments
     */
    bool containsDangerousPatterns(const std::string& arg);
    
    /**
     * Resolve command to absolute path
     */
    std::string resolveCommandPath(const std::string& command);
    
    /**
     * Create secure environment for subprocess
     */
    std::map<std::string, std::string> createSecureEnvironment(
        const std::map<std::string, std::string>& custom_env,
        bool inherit_parent);
    
    /**
     * Execute process with platform-specific implementation
     */
    ProcessResult executeProcess(const ProcessConfig& config);
    
#ifdef _WIN32
    /**
     * Windows-specific process execution
     */
    ProcessResult executeProcessWindows(const ProcessConfig& config);
    
    /**
     * Create Windows command line from arguments
     */
    std::string createWindowsCommandLine(const std::vector<std::string>& args);
#else
    /**
     * Unix-specific process execution
     */
    ProcessResult executeProcessUnix(const ProcessConfig& config);
    
    /**
     * Setup signal handlers for process management
     */
    void setupSignalHandlers();
#endif
    
    /**
     * Monitor process execution with timeout
     */
    ProcessResult monitorProcess(
#ifdef _WIN32
        HANDLE process_handle,
        HANDLE stdout_read,
        HANDLE stderr_read,
#else
        pid_t pid,
        int stdout_fd,
        int stderr_fd,
#endif
        const ProcessConfig& config);
    
    /**
     * Read output from pipes with size limits
     */
    std::string readOutput(
#ifdef _WIN32
        HANDLE pipe,
#else
        int fd,
#endif
        size_t max_size);
    
    /**
     * Log security event
     */
    void logSecurityEvent(const std::string& event, const std::string& details);
    
    /**
     * Log debug information
     */
    void logDebug(const std::string& message);
};

/**
 * Utility functions for secure subprocess management
 */
namespace subprocess_utils {
    
    /**
     * Escape argument for shell execution
     */
    std::string escapeArgument(const std::string& arg);
    
    /**
     * Check if path is within allowed directory
     */
    bool isPathWithinDirectory(const std::string& path, const std::string& base_dir);
    
    /**
     * Get executable path from command name
     */
    std::string findExecutable(const std::string& command);
    
    /**
     * Validate file permissions for execution
     */
    bool hasExecutePermission(const std::string& path);
    
    /**
     * Create temporary directory for process isolation
     */
    std::string createTempDirectory(const std::string& prefix = "secure_proc_");
    
    /**
     * Clean up temporary directory
     */
    bool cleanupTempDirectory(const std::string& path);
}

} // namespace security

#endif // SECURE_SUBPROCESS_MANAGER_H
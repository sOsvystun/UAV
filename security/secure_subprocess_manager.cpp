/**
 * Secure Subprocess Manager Implementation
 * =======================================
 * 
 * Implementation of secure subprocess execution to replace vulnerable system() calls.
 * Provides comprehensive validation, sandboxing, and monitoring capabilities.
 * 
 * Author: Security Framework Team
 */

#include "secure_subprocess_manager.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <regex>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <cstring>

#ifdef _WIN32
    #include <io.h>
    #include <fcntl.h>
    #include <shlwapi.h>
    #pragma comment(lib, "shlwapi.lib")
#else
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <errno.h>
    #include <poll.h>
#endif

namespace security {

// Static patterns for dangerous command detection
static const std::vector<std::regex> DANGEROUS_PATTERNS = {
    std::regex(R"([;&|`$(){}[\]<>*?~!])"),           // Shell metacharacters
    std::regex(R"(\.\.[\\/])"),                       // Path traversal
    std::regex(R"(\\x[0-9a-fA-F]{2})"),              // Hex encoded characters
    std::regex(R"(%[0-9a-fA-F]{2})"),                // URL encoded characters
    std::regex(R"(\\[0-7]{1,3})"),                   // Octal encoded characters
    std::regex(R"(\\\\[nt r])"),                      // Escape sequences
};

SecureSubprocessManager::SecureSubprocessManager(const SecurityPolicy& policy)
    : security_policy_(policy), debug_mode_(false) {
    
#ifndef _WIN32
    setupSignalHandlers();
#endif
    
    logDebug("SecureSubprocessManager initialized");
}

SecureSubprocessManager::~SecureSubprocessManager() {
    killAllProcesses();
    logDebug("SecureSubprocessManager destroyed");
}

ProcessResult SecureSubprocessManager::executeSecurely(const ProcessConfig& config) {
    ProcessResult result;
    
    try {
        // Validate configuration
        if (config.command.empty()) {
            result.error_message = "Command cannot be empty";
            return result;
        }
        
        // Validate command and arguments
        if (!validateCommand(config.command[0], 
                           std::vector<std::string>(config.command.begin() + 1, config.command.end()))) {
            result.error_message = "Command validation failed";
            logSecurityEvent("COMMAND_VALIDATION_FAILED", config.command[0]);
            return result;
        }
        
        // Execute the process
        result = executeProcess(config);
        
        // Log execution
        std::stringstream log_msg;
        log_msg << "Command: " << config.command[0] 
                << ", Exit Code: " << result.exit_code
                << ", Duration: " << result.execution_time.count() << "ms";
        logSecurityEvent("PROCESS_EXECUTED", log_msg.str());
        
    } catch (const std::exception& e) {
        result.error_message = std::string("Execution failed: ") + e.what();
        logSecurityEvent("EXECUTION_EXCEPTION", e.what());
    }
    
    return result;
}

ProcessResult SecureSubprocessManager::executeCommand(const std::string& command,
                                                    const std::vector<std::string>& args,
                                                    int timeout_ms) {
    ProcessConfig config;
    config.command.push_back(command);
    config.command.insert(config.command.end(), args.begin(), args.end());
    config.timeout = std::chrono::milliseconds(timeout_ms);
    
    return executeSecurely(config);
}

bool SecureSubprocessManager::validateCommand(const std::string& command, 
                                            const std::vector<std::string>& args) {
    // Check if command is in whitelist (if specified)
    if (!security_policy_.command_whitelist.empty()) {
        bool found = false;
        for (const auto& allowed : security_policy_.command_whitelist) {
            if (command.find(allowed) != std::string::npos) {
                found = true;
                break;
            }
        }
        if (!found) {
            logDebug("Command not in whitelist: " + command);
            return false;
        }
    }
    
    // Check if command is in blacklist
    for (const auto& forbidden : security_policy_.command_blacklist) {
        if (command.find(forbidden) != std::string::npos) {
            logDebug("Command in blacklist: " + command);
            return false;
        }
    }
    
    // Validate command path
    if (!validateCommandPath(command)) {
        return false;
    }
    
    // Validate arguments
    if (!validateArguments(args)) {
        return false;
    }
    
    return true;
}

bool SecureSubprocessManager::validateCommandPath(const std::string& command) {
    // Resolve to absolute path if required
    std::string resolved_command = command;
    if (security_policy_.require_absolute_paths) {
        resolved_command = resolveCommandPath(command);
        if (resolved_command.empty()) {
            logDebug("Could not resolve command path: " + command);
            return false;
        }
    }
    
    // Check if path is in allowed paths
    if (!security_policy_.allowed_paths.empty()) {
        bool path_allowed = false;
        for (const auto& allowed_path : security_policy_.allowed_paths) {
            if (resolved_command.find(allowed_path) == 0) {
                path_allowed = true;
                break;
            }
        }
        if (!path_allowed) {
            logDebug("Command path not allowed: " + resolved_command);
            return false;
        }
    }
    
    // Check if file exists and is executable
    if (!std::filesystem::exists(resolved_command)) {
        logDebug("Command file does not exist: " + resolved_command);
        return false;
    }
    
    if (!subprocess_utils::hasExecutePermission(resolved_command)) {
        logDebug("Command file is not executable: " + resolved_command);
        return false;
    }
    
    return true;
}

bool SecureSubprocessManager::validateArguments(const std::vector<std::string>& args) {
    // Check argument count
    if (args.size() > security_policy_.max_arguments) {
        logDebug("Too many arguments: " + std::to_string(args.size()));
        return false;
    }
    
    // Validate each argument
    for (const auto& arg : args) {
        // Check argument length
        if (arg.length() > security_policy_.max_argument_length) {
            logDebug("Argument too long: " + std::to_string(arg.length()));
            return false;
        }
        
        // Check for dangerous patterns
        if (containsDangerousPatterns(arg)) {
            logDebug("Dangerous pattern detected in argument: " + arg);
            return false;
        }
        
        // Check against forbidden patterns
        for (const auto& pattern : security_policy_.forbidden_patterns) {
            std::regex forbidden_regex(pattern);
            if (std::regex_search(arg, forbidden_regex)) {
                logDebug("Forbidden pattern matched: " + pattern);
                return false;
            }
        }
    }
    
    return true;
}

bool SecureSubprocessManager::containsDangerousPatterns(const std::string& arg) {
    for (const auto& pattern : DANGEROUS_PATTERNS) {
        if (std::regex_search(arg, pattern)) {
            return true;
        }
    }
    return false;
}

std::string SecureSubprocessManager::resolveCommandPath(const std::string& command) {
    // If already absolute path, return as-is
    if (std::filesystem::path(command).is_absolute()) {
        return command;
    }
    
    // Try to find in PATH
    return subprocess_utils::findExecutable(command);
}

std::string SecureSubprocessManager::sanitizePath(const std::string& path, 
                                                const std::string& base_directory) {
    try {
        // Normalize path
        std::filesystem::path normalized = std::filesystem::path(path).lexically_normal();
        
        // Check for path traversal
        std::string path_str = normalized.string();
        if (path_str.find("..") != std::string::npos) {
            logDebug("Path traversal detected: " + path);
            return "";
        }
        
        // If base directory specified, ensure path is within it
        if (!base_directory.empty()) {
            std::filesystem::path base_path = std::filesystem::path(base_directory).lexically_normal();
            std::filesystem::path full_path = base_path / normalized;
            
            if (!subprocess_utils::isPathWithinDirectory(full_path.string(), base_path.string())) {
                logDebug("Path outside base directory: " + path);
                return "";
            }
            
            return full_path.string();
        }
        
        return path_str;
        
    } catch (const std::exception& e) {
        logDebug("Path sanitization failed: " + std::string(e.what()));
        return "";
    }
}

std::vector<std::string> SecureSubprocessManager::sanitizeArguments(const std::vector<std::string>& args) {
    std::vector<std::string> sanitized;
    
    for (const auto& arg : args) {
        std::string clean_arg = arg;
        
        // Remove null bytes
        clean_arg.erase(std::remove(clean_arg.begin(), clean_arg.end(), '\\0'), clean_arg.end());
        
        // Escape special characters if needed
        clean_arg = subprocess_utils::escapeArgument(clean_arg);
        
        sanitized.push_back(clean_arg);
    }
    
    return sanitized;
}

ProcessResult SecureSubprocessManager::executeProcess(const ProcessConfig& config) {
#ifdef _WIN32
    return executeProcessWindows(config);
#else
    return executeProcessUnix(config);
#endif
}

#ifdef _WIN32
ProcessResult SecureSubprocessManager::executeProcessWindows(const ProcessConfig& config) {
    ProcessResult result;
    
    // Create pipes for stdout/stderr if needed
    HANDLE stdout_read = nullptr, stdout_write = nullptr;
    HANDLE stderr_read = nullptr, stderr_write = nullptr;
    
    if (config.capture_output) {
        SECURITY_ATTRIBUTES sa;
        sa.nLength = sizeof(SECURITY_ATTRIBUTES);
        sa.bInheritHandle = TRUE;
        sa.lpSecurityDescriptor = nullptr;
        
        if (!CreatePipe(&stdout_read, &stdout_write, &sa, 0) ||
            !CreatePipe(&stderr_read, &stderr_write, &sa, 0)) {
            result.error_message = "Failed to create pipes";
            return result;
        }
        
        // Ensure read handles are not inherited
        SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(stderr_read, HANDLE_FLAG_INHERIT, 0);
    }
    
    // Create process
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);
    
    if (config.capture_output) {
        si.hStdOutput = stdout_write;
        si.hStdError = stderr_write;
        si.dwFlags |= STARTF_USESTDHANDLES;
    }
    
    // Create command line
    std::string cmdline = createWindowsCommandLine(config.command);
    
    // Create environment block
    std::string env_block;
    auto env_map = createSecureEnvironment(config.environment, config.inherit_environment);
    for (const auto& pair : env_map) {
        env_block += pair.first + "=" + pair.second + "\\0";
    }
    env_block += "\\0";
    
    auto start_time = std::chrono::steady_clock::now();
    
    BOOL success = CreateProcessA(
        nullptr,                    // Application name
        &cmdline[0],               // Command line
        nullptr,                   // Process security attributes
        nullptr,                   // Thread security attributes
        TRUE,                      // Inherit handles
        CREATE_NO_WINDOW,          // Creation flags
        env_block.empty() ? nullptr : &env_block[0], // Environment
        config.working_directory.empty() ? nullptr : config.working_directory.c_str(), // Current directory
        &si,                       // Startup info
        &pi                        // Process info
    );
    
    if (config.capture_output) {
        CloseHandle(stdout_write);
        CloseHandle(stderr_write);
    }
    
    if (!success) {
        result.error_message = "Failed to create process";
        if (stdout_read) CloseHandle(stdout_read);
        if (stderr_read) CloseHandle(stderr_read);
        return result;
    }
    
    // Monitor process
    {
        std::lock_guard<std::mutex> lock(processes_mutex_);
        process_handles_.push_back(pi.hProcess);
    }
    
    result = monitorProcess(pi.hProcess, stdout_read, stderr_read, config);
    
    auto end_time = std::chrono::steady_clock::now();
    result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Cleanup
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    if (stdout_read) CloseHandle(stdout_read);
    if (stderr_read) CloseHandle(stderr_read);
    
    {
        std::lock_guard<std::mutex> lock(processes_mutex_);
        process_handles_.erase(
            std::remove(process_handles_.begin(), process_handles_.end(), pi.hProcess),
            process_handles_.end());
    }
    
    return result;
}

std::string SecureSubprocessManager::createWindowsCommandLine(const std::vector<std::string>& args) {
    std::string cmdline;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) cmdline += " ";
        
        // Quote argument if it contains spaces
        std::string arg = args[i];
        if (arg.find(' ') != std::string::npos) {
            cmdline += "\\"" + arg + "\\"";
        } else {
            cmdline += arg;
        }
    }
    return cmdline;
}

#else // Unix implementation

ProcessResult SecureSubprocessManager::executeProcessUnix(const ProcessConfig& config) {
    ProcessResult result;
    
    // Create pipes for stdout/stderr if needed
    int stdout_pipe[2] = {-1, -1};
    int stderr_pipe[2] = {-1, -1};
    
    if (config.capture_output) {
        if (pipe(stdout_pipe) == -1 || pipe(stderr_pipe) == -1) {
            result.error_message = "Failed to create pipes";
            return result;
        }
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Fork process
    pid_t pid = fork();
    if (pid == -1) {
        result.error_message = "Failed to fork process";
        if (config.capture_output) {
            close(stdout_pipe[0]); close(stdout_pipe[1]);
            close(stderr_pipe[0]); close(stderr_pipe[1]);
        }
        return result;
    }
    
    if (pid == 0) {
        // Child process
        
        // Setup pipes
        if (config.capture_output) {
            close(stdout_pipe[0]); // Close read end
            close(stderr_pipe[0]); // Close read end
            
            dup2(stdout_pipe[1], STDOUT_FILENO);
            dup2(stderr_pipe[1], STDERR_FILENO);
            
            close(stdout_pipe[1]);
            close(stderr_pipe[1]);
        }
        
        // Change working directory if specified
        if (!config.working_directory.empty()) {
            if (chdir(config.working_directory.c_str()) != 0) {
                _exit(127);
            }
        }
        
        // Setup environment
        auto env_map = createSecureEnvironment(config.environment, config.inherit_environment);
        std::vector<std::string> env_strings;
        std::vector<char*> env_ptrs;
        
        for (const auto& pair : env_map) {
            env_strings.push_back(pair.first + "=" + pair.second);
        }
        
        for (const auto& env_str : env_strings) {
            env_ptrs.push_back(const_cast<char*>(env_str.c_str()));
        }
        env_ptrs.push_back(nullptr);
        
        // Prepare arguments
        std::vector<char*> argv;
        for (const auto& arg : config.command) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        argv.push_back(nullptr);
        
        // Execute
        execve(config.command[0].c_str(), argv.data(), env_ptrs.data());
        
        // If we get here, exec failed
        _exit(127);
    }
    
    // Parent process
    if (config.capture_output) {
        close(stdout_pipe[1]); // Close write end
        close(stderr_pipe[1]); // Close write end
    }
    
    {
        std::lock_guard<std::mutex> lock(processes_mutex_);
        process_pids_.push_back(pid);
    }
    
    result = monitorProcess(pid, 
                          config.capture_output ? stdout_pipe[0] : -1,
                          config.capture_output ? stderr_pipe[0] : -1,
                          config);
    
    auto end_time = std::chrono::steady_clock::now();
    result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Cleanup
    if (config.capture_output) {
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
    }
    
    {
        std::lock_guard<std::mutex> lock(processes_mutex_);
        process_pids_.erase(
            std::remove(process_pids_.begin(), process_pids_.end(), pid),
            process_pids_.end());
    }
    
    return result;
}

void SecureSubprocessManager::setupSignalHandlers() {
    // Setup signal handlers for clean process termination
    signal(SIGCHLD, SIG_DFL);
}

#endif

ProcessResult SecureSubprocessManager::monitorProcess(
#ifdef _WIN32
    HANDLE process_handle,
    HANDLE stdout_read,
    HANDLE stderr_read,
#else
    pid_t pid,
    int stdout_fd,
    int stderr_fd,
#endif
    const ProcessConfig& config) {
    
    ProcessResult result;
    
#ifdef _WIN32
    // Wait for process with timeout
    DWORD wait_result = WaitForSingleObject(process_handle, static_cast<DWORD>(config.timeout.count()));
    
    if (wait_result == WAIT_TIMEOUT) {
        result.timed_out = true;
        TerminateProcess(process_handle, 1);
        result.error_message = "Process timed out";
    } else if (wait_result == WAIT_OBJECT_0) {
        DWORD exit_code;
        if (GetExitCodeProcess(process_handle, &exit_code)) {
            result.exit_code = static_cast<int>(exit_code);
            result.success = (exit_code == 0);
        }
    } else {
        result.error_message = "Process wait failed";
    }
    
    // Read output
    if (config.capture_output && stdout_read) {
        result.stdout_output = readOutput(stdout_read, config.max_output_size);
    }
    if (config.capture_output && stderr_read) {
        result.stderr_output = readOutput(stderr_read, config.max_output_size);
    }
    
#else
    // Monitor process with timeout
    auto timeout_end = std::chrono::steady_clock::now() + config.timeout;
    int status;
    
    while (std::chrono::steady_clock::now() < timeout_end) {
        pid_t wait_result = waitpid(pid, &status, WNOHANG);
        
        if (wait_result == pid) {
            // Process finished
            if (WIFEXITED(status)) {
                result.exit_code = WEXITSTATUS(status);
                result.success = (result.exit_code == 0);
            } else if (WIFSIGNALED(status)) {
                result.exit_code = -WTERMSIG(status);
                result.error_message = "Process terminated by signal";
            }
            break;
        } else if (wait_result == -1) {
            result.error_message = "Process wait failed";
            break;
        }
        
        // Sleep briefly before checking again
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Check if process timed out
    if (std::chrono::steady_clock::now() >= timeout_end) {
        result.timed_out = true;
        kill(pid, SIGTERM);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        kill(pid, SIGKILL);
        waitpid(pid, nullptr, 0); // Clean up zombie
        result.error_message = "Process timed out";
    }
    
    // Read output
    if (config.capture_output && stdout_fd != -1) {
        result.stdout_output = readOutput(stdout_fd, config.max_output_size);
    }
    if (config.capture_output && stderr_fd != -1) {
        result.stderr_output = readOutput(stderr_fd, config.max_output_size);
    }
#endif
    
    return result;
}

std::string SecureSubprocessManager::readOutput(
#ifdef _WIN32
    HANDLE pipe,
#else
    int fd,
#endif
    size_t max_size) {
    
    std::string output;
    char buffer[4096];
    
#ifdef _WIN32
    DWORD bytes_read;
    while (ReadFile(pipe, buffer, sizeof(buffer) - 1, &bytes_read, nullptr) && bytes_read > 0) {
        buffer[bytes_read] = '\\0';
        output += buffer;
        
        if (output.size() > max_size) {
            output = output.substr(0, max_size);
            output += "\\n[Output truncated - size limit exceeded]";
            break;
        }
    }
#else
    ssize_t bytes_read;
    while ((bytes_read = read(fd, buffer, sizeof(buffer) - 1)) > 0) {
        buffer[bytes_read] = '\\0';
        output += buffer;
        
        if (output.size() > max_size) {
            output = output.substr(0, max_size);
            output += "\\n[Output truncated - size limit exceeded]";
            break;
        }
    }
#endif
    
    return output;
}

std::map<std::string, std::string> SecureSubprocessManager::createSecureEnvironment(
    const std::map<std::string, std::string>& custom_env,
    bool inherit_parent) {
    
    std::map<std::string, std::string> env;
    
    if (inherit_parent) {
        // Copy safe environment variables from parent
        const char* safe_vars[] = {
            "PATH", "HOME", "USER", "LANG", "LC_ALL", "TZ", "TMPDIR", "TEMP"
        };
        
        for (const char* var : safe_vars) {
            const char* value = std::getenv(var);
            if (value) {
                env[var] = value;
            }
        }
    } else {
        // Minimal secure environment
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin";
#ifdef _WIN32
        env["SYSTEMROOT"] = "C:\\\\Windows";
        env["TEMP"] = "C:\\\\Temp";
#else
        env["HOME"] = "/tmp";
        env["TMPDIR"] = "/tmp";
#endif
    }
    
    // Add custom environment variables
    for (const auto& pair : custom_env) {
        env[pair.first] = pair.second;
    }
    
    return env;
}

void SecureSubprocessManager::killAllProcesses() {
    std::lock_guard<std::mutex> lock(processes_mutex_);
    
#ifdef _WIN32
    for (HANDLE handle : process_handles_) {
        TerminateProcess(handle, 1);
    }
    process_handles_.clear();
#else
    for (pid_t pid : process_pids_) {
        kill(pid, SIGTERM);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        kill(pid, SIGKILL);
    }
    process_pids_.clear();
#endif
}

size_t SecureSubprocessManager::getRunningProcessCount() const {
    std::lock_guard<std::mutex> lock(processes_mutex_);
#ifdef _WIN32
    return process_handles_.size();
#else
    return process_pids_.size();
#endif
}

void SecureSubprocessManager::updateSecurityPolicy(const SecurityPolicy& policy) {
    security_policy_ = policy;
    logDebug("Security policy updated");
}

void SecureSubprocessManager::setDebugMode(bool enabled) {
    debug_mode_ = enabled;
}

void SecureSubprocessManager::logSecurityEvent(const std::string& event, const std::string& details) {
    // TODO: Integrate with audit logging system
    std::cout << "[SECURITY] " << event << ": " << details << std::endl;
}

void SecureSubprocessManager::logDebug(const std::string& message) {
    if (debug_mode_) {
        std::cout << "[DEBUG] " << message << std::endl;
    }
}

// Utility functions implementation
namespace subprocess_utils {

std::string escapeArgument(const std::string& arg) {
    std::string escaped = arg;
    
    // Escape quotes
    size_t pos = 0;
    while ((pos = escaped.find('"', pos)) != std::string::npos) {
        escaped.insert(pos, "\\\\");
        pos += 2;
    }
    
    return escaped;
}

bool isPathWithinDirectory(const std::string& path, const std::string& base_dir) {
    try {
        std::filesystem::path abs_path = std::filesystem::absolute(path);
        std::filesystem::path abs_base = std::filesystem::absolute(base_dir);
        
        auto rel_path = std::filesystem::relative(abs_path, abs_base);
        return !rel_path.empty() && rel_path.native()[0] != '.';
        
    } catch (const std::exception&) {
        return false;
    }
}

std::string findExecutable(const std::string& command) {
    const char* path_env = std::getenv("PATH");
    if (!path_env) return "";
    
    std::string path_str(path_env);
    std::istringstream path_stream(path_str);
    std::string path_dir;
    
#ifdef _WIN32
    const char delimiter = ';';
    const std::string exe_ext = ".exe";
#else
    const char delimiter = ':';
    const std::string exe_ext = "";
#endif
    
    while (std::getline(path_stream, path_dir, delimiter)) {
        std::filesystem::path full_path = std::filesystem::path(path_dir) / (command + exe_ext);
        
        if (std::filesystem::exists(full_path) && hasExecutePermission(full_path.string())) {
            return full_path.string();
        }
    }
    
    return "";
}

bool hasExecutePermission(const std::string& path) {
#ifdef _WIN32
    // On Windows, check if file exists and has .exe extension or is in PATH
    return std::filesystem::exists(path);
#else
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return false;
    }
    
    return (st.st_mode & S_IXUSR) || (st.st_mode & S_IXGRP) || (st.st_mode & S_IXOTH);
#endif
}

std::string createTempDirectory(const std::string& prefix) {
    try {
        std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
        std::filesystem::path unique_dir = temp_dir / (prefix + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        
        std::filesystem::create_directories(unique_dir);
        return unique_dir.string();
        
    } catch (const std::exception&) {
        return "";
    }
}

bool cleanupTempDirectory(const std::string& path) {
    try {
        return std::filesystem::remove_all(path) > 0;
    } catch (const std::exception&) {
        return false;
    }
}

} // namespace subprocess_utils

} // namespace security
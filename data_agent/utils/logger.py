import os
import time
import traceback


class CustomLogger:
    def __init__(
        self, log_file="custom_agent_log.log", console_level="INFO", file_level="DEBUG"
    ):
        self.log_file = log_file
        self.console_level = self._get_level_value(console_level)
        self.file_level = self._get_level_value(file_level)
        self._ensure_log_file_exists()

    def _get_level_value(self, level_name):
        levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
        return levels.get(
            level_name.upper(), 0
        )  # Default to lowest level if name is unknown

    def _ensure_log_file_exists(self):
        """Ensure the log file exists upon initialization."""
        if not os.path.exists(self.log_file):
            try:
                with open(self.log_file, "w") as f:
                    f.write(
                        f"Log file created on {time.strftime('%Y-%m-%d %H:%M:%S%z')}\n"
                    )
            except IOError as e:
                print(f"ERROR: Could not create log file {self.log_file}: {e}")

    def _log(self, level, message, exc_info=False):
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        level_name = level.upper()
        log_message = f"{timestamp} - {level_name} - {message}"

        # Add traceback if exc_info is True and an exception is active
        if exc_info:
            exc_type, exc_value, exc_traceback = traceback.sys.exc_info()
            if exc_traceback:
                log_message += f"\n{traceback.format_exc()}"

        # Write to file if level is high enough
        if self._get_level_value(level_name) >= self.file_level:
            try:
                with open(self.log_file, "a") as f:
                    f.write(log_message + "\n")
            except IOError as e:
                print(f"ERROR: Could not write to log file {self.log_file}: {e}")

        # Print to console if level is high enough
        if self._get_level_value(level_name) >= self.console_level:
            print(log_message)

    def debug(self, message, exc_info=False):
        self._log("DEBUG", message, exc_info)

    def info(self, message, exc_info=False):
        self._log("INFO", message, exc_info)

    def warning(self, message, exc_info=False):
        self._log("WARNING", message, exc_info)

    def error(self, message, exc_info=False):
        self._log("ERROR", message, exc_info)

    def critical(self, message, exc_info=False):
        self._log("CRITICAL", message, exc_info)


# Example usage:
# Instantiate with desired levels
# logger = CustomLogger(console_level="INFO", file_level="DEBUG")

# Use the logger methods
# logger.debug("This is a debug message.") # Only in file
# logger.info("This is an info message.")   # In file and console
# logger.error("This is an error message.") # In file and console

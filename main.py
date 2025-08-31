import logging
from agents.data_describer import describe_agent, terminate_agent
from utils.logger import CustomLogger

logger = CustomLogger(console_level="INFO", file_level="DEBUG")



if __name__ == '__main__':
    logger.info("Starting agent loop.")
    # Run the agent with user input
    user_input = "Describe the dataframes, after listing them in the directory."
    final_memory = describe_agent.run(user_input)

    # Print the final memory
    logger.debug(final_memory.get_memories())
    logger.info("Ending Agent loop.")
from data_agent.agents.data_scientist import data_scientist, user_input
from data_agent.utils.logger import CustomLogger

logger = CustomLogger(console_level="INFO", file_level="DEBUG")


if __name__ == "__main__":
    logger.info("Starting agent loop.")
    # Run the agent with user input
    final_memory = data_scientist.run(user_input)

    # Print the final memory
    logger.debug(final_memory.get_memories())
    logger.info("Ending Agent loop.")

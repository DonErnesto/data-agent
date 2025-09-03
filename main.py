import logging
from data_agent.agents.data_describer import describe_agent, terminate_agent
from data_agent.agents.diff_agent import diff_agent, user_input
from data_agent.utils.logger import CustomLogger

logger = CustomLogger(console_level="INFO", file_level="DEBUG")



if __name__ == '__main__':
    logger.info("Starting agent loop.")
    # Run the agent with user input
    final_memory = diff_agent.run(user_input)

    # Print the final memory
    logger.debug(final_memory.get_memories())
    logger.info("Ending Agent loop.")
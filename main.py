import logging
from agents.data_describer import describe_agent, terminate_agent
from agents.qa_agent import qa_agent, user_input
from utils.logger import CustomLogger

logger = CustomLogger(console_level="INFO", file_level="DEBUG")



if __name__ == '__main__':
    logger.info("Starting agent loop.")
    # Run the agent with user input
    final_memory = qa_agent.run(user_input)

    # Print the final memory
    logger.debug(final_memory.get_memories())
    logger.info("Ending Agent loop.")
from agents.data_describer import describe_agent

if __name__ == '__main__':
    # Run the agent with user input
    user_input = "Describe the data in this directory."
    final_memory = describe_agent.run(user_input)

    # Print the final memory
    print(final_memory.get_memories())
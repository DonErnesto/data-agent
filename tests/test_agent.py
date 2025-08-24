from src.agent.agent import Agent

def test_agent_step():
    agent = Agent(goals=["test"])
    result = agent.step()
    assert isinstance(result, str)


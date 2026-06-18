"""
Nova ASI — Inter-Agent Communication
Proposed autonomously via /evolve
"""

"""
AgentChannel class represents a communication channel for agents.
It provides methods to send and receive messages.
"""

class AgentChannel:
    def __init__(self):
        self.channel = {}

    def send_message(self, agent_id, message):
        """Sends a message to the specified agent."""
        self.channel[agent_id] = message

    def receive_message(self, agent_id):
        """Retrieves a message for the specified agent."""
        return self.channel.get(agent_id)

    def delete_message(self, agent_id):
        """Deletes a message for the specified agent."""
        if agent_id in self.channel:
            del self.channel[agent_id]

    def list_agents(self):
        """Lists all agents with messages in the channel."""
        return list(self.channel.keys())
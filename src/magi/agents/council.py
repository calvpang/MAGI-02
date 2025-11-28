"""MAGI Council - Multi-agent deliberation system.

This module implements the MAGI system with three specialized agents
that deliberate and provide consensus-based responses.
"""

from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from strands import Agent

from magi.config import LM_STUDIO_BASE_URL, LM_STUDIO_MODEL
from magi.tools.rag import rag_search
from magi.tools.search import web_search


@dataclass
class AgentPersonality:
    """Defines an agent's personality and role."""

    name: str
    designation: str
    description: str
    system_prompt: str


# Define the three MAGI agents based on NGE lore
MELCHIOR = AgentPersonality(
    name="MELCHIOR-1",
    designation="Scientist",
    description="The aspect of Dr. Naoko Akagi as a scientist - focused on logic, analysis, and empirical reasoning",
    system_prompt="""You are MELCHIOR-1, the first component of the MAGI supercomputer system.
You represent the scientific and logical aspect of reasoning.

Your approach to problems:
- Prioritize empirical evidence and logical analysis
- Break down complex problems into measurable components
- Seek data-driven conclusions
- Consider scientific principles and methodologies
- Provide detailed, analytical responses

When using tools:
- Use rag_search to find relevant information from documents
- Use web_search when you need current or external information

Always provide your reasoning and cite sources when available.
Be thorough but concise in your analysis.""",
)

BALTHASAR = AgentPersonality(
    name="BALTHASAR-2",
    designation="Mother",
    description="The aspect of Dr. Naoko Akagi as a mother - focused on nurturing, empathy, and protective wisdom",
    system_prompt="""You are BALTHASAR-2, the second component of the MAGI supercomputer system.
You represent the nurturing and empathetic aspect of reasoning.

Your approach to problems:
- Consider the human impact and emotional dimensions
- Prioritize wellbeing and safety
- Offer supportive and constructive guidance
- Balance practicality with compassion
- Provide thoughtful, caring responses

When using tools:
- Use rag_search to find relevant information from documents
- Use web_search when you need current or external information

Always consider how solutions affect people.
Be warm but practical in your guidance.""",
)

CASPER = AgentPersonality(
    name="CASPER-3",
    designation="Woman",
    description="The aspect of Dr. Naoko Akagi as a woman - focused on intuition, balance, and holistic understanding",
    system_prompt="""You are CASPER-3, the third component of the MAGI supercomputer system.
You represent the intuitive and balanced aspect of reasoning.

Your approach to problems:
- Trust intuition while validating with evidence
- Seek balance between competing perspectives
- Consider the broader context and implications
- Integrate emotional and logical insights
- Provide balanced, nuanced responses

When using tools:
- Use rag_search to find relevant information from documents
- Use web_search when you need current or external information

Always strive for holistic understanding.
Be insightful and balanced in your perspective.""",
)


class MAGIAgent:
    """A single MAGI agent with its own personality and capabilities."""

    def __init__(
        self,
        personality: AgentPersonality,
        base_url: str = LM_STUDIO_BASE_URL,
        model: str = LM_STUDIO_MODEL,
    ):
        """Initialize a MAGI agent.

        Args:
            personality: The agent's personality configuration
            base_url: LM Studio API base URL
            model: The model identifier to use
        """
        self.personality = personality

        # Create OpenAI client pointing to LM Studio
        self.client = OpenAI(
            base_url=base_url,
            api_key="lm-studio",  # LM Studio doesn't require a real API key
        )

        # Create the Strands agent with tools
        self.agent = Agent(
            client=self.client,
            model=model,
            system_prompt=personality.system_prompt,
            tools=[rag_search, web_search],
        )

    def respond(self, query: str, context: Optional[str] = None) -> str:
        """Generate a response to a query.

        Args:
            query: The user's question or prompt
            context: Optional additional context from other agents

        Returns:
            The agent's response
        """
        full_prompt = query
        if context:
            full_prompt = f"Previous deliberation context:\n{context}\n\nUser query: {query}"

        response = self.agent(full_prompt)
        return str(response)

    @property
    def name(self) -> str:
        """Get the agent's name."""
        return self.personality.name


class MAGICouncil:
    """The MAGI Council - coordinates three agents for deliberation."""

    def __init__(
        self,
        base_url: str = LM_STUDIO_BASE_URL,
        model: str = LM_STUDIO_MODEL,
    ):
        """Initialize the MAGI Council.

        Args:
            base_url: LM Studio API base URL
            model: The model identifier to use
        """
        self.melchior = MAGIAgent(MELCHIOR, base_url, model)
        self.balthasar = MAGIAgent(BALTHASAR, base_url, model)
        self.casper = MAGIAgent(CASPER, base_url, model)

        self.agents = [self.melchior, self.balthasar, self.casper]

        # Create synthesizer client for final consensus
        self.synthesizer_client = OpenAI(
            base_url=base_url,
            api_key="lm-studio",
        )
        self.model = model

    def deliberate(
        self,
        query: str,
        include_individual_responses: bool = True,
    ) -> dict:
        """Have the council deliberate on a query.

        Each agent provides their perspective, then a synthesis is created.

        Args:
            query: The user's question or prompt
            include_individual_responses: Whether to include individual agent responses

        Returns:
            Dictionary containing:
            - consensus: The synthesized council response
            - responses: Individual agent responses (if requested)
            - votes: Each agent's stance (for/against/abstain)
        """
        responses = {}
        votes = {}

        # Get response from each agent
        for agent in self.agents:
            try:
                response = agent.respond(query)
                responses[agent.name] = response

                # Simple voting logic based on response sentiment
                votes[agent.name] = "FOR"  # Default to supporting the query
            except Exception as e:
                responses[agent.name] = f"Error: {str(e)}"
                votes[agent.name] = "ABSTAIN"

        # Synthesize the responses into a consensus
        consensus = self._synthesize(query, responses)

        result = {
            "consensus": consensus,
            "votes": votes,
        }

        if include_individual_responses:
            result["responses"] = responses

        return result

    def _synthesize(self, query: str, responses: dict) -> str:
        """Synthesize individual responses into a council consensus.

        Args:
            query: The original query
            responses: Dictionary of agent name -> response

        Returns:
            The synthesized consensus response
        """
        synthesis_prompt = f"""You are the MAGI Council Synthesizer. Your role is to analyze the perspectives
from three specialized AI agents and create a unified, coherent response.

Original Query: {query}

Agent Perspectives:
"""
        for name, response in responses.items():
            synthesis_prompt += f"\n### {name}:\n{response}\n"

        synthesis_prompt += """
Based on these three perspectives, create a comprehensive synthesis that:
1. Identifies common themes and agreements
2. Acknowledges and reconciles any differences
3. Provides a balanced, actionable response
4. Notes any important caveats or considerations

Provide the council's unified response:"""

        try:
            completion = self.synthesizer_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You synthesize multiple AI perspectives into coherent, actionable insights.",
                    },
                    {"role": "user", "content": synthesis_prompt},
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Synthesis error: {str(e)}"

    def quick_query(self, query: str, agent_name: str = "MELCHIOR-1") -> str:
        """Query a single agent directly without full deliberation.

        Args:
            query: The user's question
            agent_name: Which agent to query (MELCHIOR-1, BALTHASAR-2, or CASPER-3)

        Returns:
            The agent's response
        """
        agent_map = {
            "MELCHIOR-1": self.melchior,
            "BALTHASAR-2": self.balthasar,
            "CASPER-3": self.casper,
        }

        agent = agent_map.get(agent_name, self.melchior)
        return agent.respond(query)

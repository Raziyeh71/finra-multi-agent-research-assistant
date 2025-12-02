"""
LLM-based Research Planner Agent
Decides search strategy based on user goals
"""

import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import openai


class ResearchPlan(BaseModel):
    """Structured research plan output"""
    search_terms: List[str] = Field(description="Optimized search terms for papers/videos")
    sources: List[str] = Field(description="Sources to search: arxiv, nature, ieee, youtube, web")
    max_papers: int = Field(description="Number of papers to retrieve", default=10)
    max_age_days: int = Field(description="Recency filter in days", default=90)
    focus_areas: List[str] = Field(description="Specific focus areas")
    reasoning: str = Field(description="Explanation of the plan")


class PlannerAgent:
    """
    LLM-based planner that creates research strategy from user goals.
    
    This agent analyzes the user's request and generates an optimal
    research plan including search terms, sources, and filters.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize the planner agent.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for planning (default: gpt-4)
        """
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key
    
    def create_plan(self, user_goal: str, current_year: int = 2024) -> ResearchPlan:
        """
        Create research plan from user goal.
        
        Args:
            user_goal: User's research objective
            current_year: Current year for recency filtering
            
        Returns:
            ResearchPlan with optimized strategy
        """
        
        system_prompt = f"""You are an expert research planning agent for finance/trading AI research.

Given a user's research goal, create an optimal search strategy.

Consider:
1. What specific search terms will find the most relevant papers?
2. Which sources are best? (arxiv for preprints, nature/ieee for peer-reviewed, youtube for tutorials, web for code/blogs)
3. How recent should the content be? (30 days for breaking news, 90 days for recent research, 180+ for comprehensive surveys)
4. What are the key focus areas? (e.g., deep learning, reinforcement learning, LLMs, risk management)

Current year: {current_year}

Output a JSON object with this exact structure:
{{
  "search_terms": ["term1", "term2", "term3"],
  "sources": ["arxiv", "nature", "youtube", "web"],
  "max_papers": 10,
  "max_age_days": 90,
  "focus_areas": ["area1", "area2"],
  "reasoning": "Brief explanation of the strategy"
}}

IMPORTANT: 
- Search terms should be specific and actionable
- Exclude crypto/cryptocurrency unless explicitly requested
- Focus on practical, implementation-oriented research
- Prefer recent work (last 30-90 days) unless user wants comprehensive survey
"""

        user_prompt = f"""User Research Goal:
{user_goal}

Create an optimal research plan (output only valid JSON):"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON if wrapped in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            plan_dict = json.loads(content)
            plan = ResearchPlan(**plan_dict)
            
            print(f"\n🧠 Research Plan Created:")
            print(f"   Search Terms: {plan.search_terms}")
            print(f"   Sources: {plan.sources}")
            print(f"   Recency: Last {plan.max_age_days} days")
            print(f"   Reasoning: {plan.reasoning}\n")
            
            return plan
            
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse LLM response as JSON: {e}")
            print(f"Raw response: {content}")
            # Fallback to default plan
            return self._create_default_plan(user_goal)
        except Exception as e:
            print(f"Warning: Planning failed: {e}")
            return self._create_default_plan(user_goal)
    
    def _create_default_plan(self, user_goal: str) -> ResearchPlan:
        """Fallback plan if LLM planning fails"""
        return ResearchPlan(
            search_terms=[
                "AI algorithmic trading",
                "machine learning stock prediction",
                "deep learning financial markets"
            ],
            sources=["arxiv", "nature", "ieee"],
            max_papers=5,
            max_age_days=180,
            focus_areas=["machine learning", "algorithmic trading"],
            reasoning="Default fallback plan due to planning error"
        )

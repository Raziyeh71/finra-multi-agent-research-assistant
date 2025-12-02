#!/usr/bin/env python3
"""
FinRA Multi-Agent Research System - CLI Runner

Usage:
    python run_multi_agent.py "Find latest LLM trading papers"
    python run_multi_agent.py --goal "AI stock prediction" --max-age 60
    python run_multi_agent.py --help

Only requires OpenAI API key - all other tools are FREE!
"""

import asyncio
import argparse
import os
import sys

from dotenv import load_dotenv


async def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="FinRA Multi-Agent Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_multi_agent.py "Find latest LLM trading papers"
  python run_multi_agent.py --goal "AI stock prediction" --max-age 60 --include-videos
  python run_multi_agent.py --goal "transformer models finance" --output results.json

Only requires OpenAI API key - all other tools are FREE!
Set OPENAI_API_KEY in .env file or environment variable.
        """
    )
    
    parser.add_argument(
        "goal",
        nargs="?",
        help="Research goal (e.g., 'Find latest LLM trading papers')"
    )
    parser.add_argument(
        "--goal", "-g",
        dest="goal_flag",
        help="Research goal (alternative to positional argument)"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--max-age", "-a",
        type=int,
        default=90,
        help="Maximum age of papers in days (default: 90)"
    )
    parser.add_argument(
        "--max-papers", "-p",
        type=int,
        default=10,
        help="Maximum papers to summarize (default: 10)"
    )
    parser.add_argument(
        "--include-videos", "-v",
        action="store_true",
        help="Include YouTube video search"
    )
    parser.add_argument(
        "--include-web", "-w",
        action="store_true",
        default=True,
        help="Include web search (default: True)"
    )
    parser.add_argument(
        "--output", "-o",
        default="finra_research_results.json",
        help="Output JSON file (default: finra_research_results.json)"
    )
    parser.add_argument(
        "--memory-path", "-m",
        default="./finra_memory",
        help="Path for memory storage (default: ./finra_memory)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Get goal from either positional or flag argument
    goal = args.goal or args.goal_flag
    
    if not goal:
        parser.print_help()
        print("\n❌ Error: Please provide a research goal")
        print("Example: python run_multi_agent.py \"Find latest LLM trading papers\"")
        sys.exit(1)
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ Error: OpenAI API key required")
        print("Set OPENAI_API_KEY in .env file or use --api-key flag")
        sys.exit(1)
    
    # Import and run
    try:
        from multi_agent_system import FinRAMultiAgent
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    
    # Create agent
    agent = FinRAMultiAgent(
        openai_api_key=api_key,
        memory_path=args.memory_path,
        max_papers_to_summarize=args.max_papers,
        include_videos=args.include_videos,
        include_web=args.include_web,
        debug=args.debug,
    )
    
    # Run research
    try:
        results = await agent.research(goal, max_age_days=args.max_age)
        
        # Save results
        agent.save_results(results, args.output)
        
        print(f"\n✅ Research complete! Results saved to {args.output}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Research interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

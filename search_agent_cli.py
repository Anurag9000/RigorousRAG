import argparse
import sys
import os
from search_agent import SearchAgent
from tools.models import AgentAnswer

def main():
    parser = argparse.ArgumentParser(description="Academic Agentic Search CLI")
    parser.add_argument("--query", "-q", type=str, help="Run a single query and exit.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use.")
    
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running the agent.")
        sys.exit(1)

    agent = SearchAgent(model=args.model, api_key=api_key)

    if args.query:
        # One-shot mode
        print(f"Agent: Analyzing '{args.query}'...\n")
        result = agent.run(args.query)
        print_result(result)
    else:
        # Interactive mode
        print("Academic Search Agent (type 'exit' or 'quit' to stop)")
        print("-----------------------------------------------------")
        while True:
            try:
                user_input = input("You> ").strip()
                if user_input.lower() in ("exit", "quit"):
                    break
                if not user_input:
                    continue
                
                print("\nAgent: Thinking...")
                result = agent.run(user_input)
                print_result(result)
                print("-" * 40)
            except KeyboardInterrupt:
                print("\nExiting...")
                break

def print_result(result: AgentAnswer):
    print("\nAnswer:")
    print(result.answer)
    
    if result.citations:
        print("\nCitations:")
        for cit in result.citations:
            # Format: [1] Title (Source Type) - URL
            #         Snippet...
            print(f"{cit.label} {cit.title} ({cit.source_type})")
            print(f"    URL: {cit.url}")
            if cit.snippet:
                snippet = cit.snippet.replace('\n', ' ')
                if len(snippet) > 100:
                    snippet = snippet[:97] + "..."
                print(f"    Excerpt: {snippet}")
            print()

if __name__ == "__main__":
    main()

import argparse
import sys
import os
from search_agent import SearchAgent
from tools.models import AgentAnswer

def main():
    parser = argparse.ArgumentParser(description="Academic Agentic Search CLI")
    parser.add_argument("--query", "-q", type=str, help="Run a single query and exit.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use (default: gpt-4o).")
    parser.add_argument("--local", action="store_true", help="Run 100% locally using Ollama (default model: llama3.1).")
    parser.add_argument("--demo", action="store_true", help="Run ultra-fast demo mode with a tiny model (qwen2.5:1.5b).")
    
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if args.local or args.demo:
        # Override settings for local Ollama instance
        mode_name = "DEMO" if args.demo else "LOCAL"
        print(f"[INFO] Running in {mode_name} mode via Ollama...")
        api_key = "ollama"
        base_url = "http://localhost:11434/v1"
        if args.demo:
            args.model = "qwen2.5:0.5b"
        elif args.model == "gpt-4o":
            args.model = "llama3.1"
    else:
        if not api_key and not base_url:
            print("Error: OPENAI_API_KEY environment variable not set.")
            print("Please set it before running the agent, or use the --local flag.")
            sys.exit(1)

    print(f"[INFO] Initializing Agent with model: {args.model}")
    agent = SearchAgent(model=args.model, api_key=api_key, base_url=base_url)

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

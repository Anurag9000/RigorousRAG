import argparse
import sys
import json
from pathlib import Path
from tools.ingestion import ingest_file

def main():
    parser = argparse.ArgumentParser(description="Ingest documents (PDF, Word, Text) into the system.")
    parser.add_argument("paths", nargs="+", help="Files or directories to ingest.")
    parser.add_argument("--output", "-o", help="Output JSON file for ingested data.", default="ingested_docs.json")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursively ingest directories.")
    
    args = parser.parse_args()
    
    files_to_process = []
    
    for p_str in args.paths:
        path = Path(p_str)
        if path.is_file():
            files_to_process.append(path)
        elif path.is_dir():
             pattern = "**/*" if args.recursive else "*"
             for child in path.glob(pattern):
                 if child.is_file():
                     files_to_process.append(child)
    
    print(f"Found {len(files_to_process)} files to process.")
    
    # Initialize RAG layer
    try:
        from tools.rag import get_rag_layer
        rag_layer = get_rag_layer()
        print("RAG Layer initialized.")
    except Exception as e:
        print(f"Warning: RAG Layer could not be initialized. Indexing will be skipped. Error: {e}")
        rag_layer = None

    results = []
    success_count = 0
    
    for file_path in files_to_process:
        print(f"Ingesting: {file_path} ...", end=" ", flush=True)
        result = ingest_file(str(file_path))
        if result.success:
            print("OK", end=" ")
            if rag_layer and result.document:
                try:
                    # Goal 19: Pre-generated short summary
                    # Placeholder: in full implementation, call LLM to summarize
                    summary = result.document.text[:500] + "..." 
                    
                    rag_layer.add_document(
                        doc_id=result.document.id,
                        text=result.document.text,
                        metadata={
                            "filename": result.document.filename,
                            "mime_type": result.document.mime_type,
                            "summary": summary
                        }
                    )
                    print("(Indexed + Summary)", end=" ")
                except Exception as e:
                    print(f"(Index Failed: {e})", end=" ")
            print("")
            success_count += 1
            results.append(result.document.model_dump())
        else:
            print(f"FAILED ({result.error})")
            
    print(f"\nIngestion Complete. {success_count}/{len(files_to_process)} successful.")
    
    if args.output and results:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()

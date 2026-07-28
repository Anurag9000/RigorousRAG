import os
import subprocess
import sys


def test_all_public_modules_import_together_in_clean_process(tmp_path):
    environment = os.environ.copy()
    environment.update(
        {
            "UPLOAD_DIR": str(tmp_path / "uploads"),
            "JOB_DB_PATH": str(tmp_path / "jobs.sqlite3"),
            "DOCUMENT_DB_PATH": str(tmp_path / "documents.sqlite3"),
            "CHROMA_PATH": str(tmp_path / "vectors"),
            "CLASSIC_STORAGE_DIR": str(tmp_path / "classic"),
            "ORPHAN_CLEANUP_ON_STARTUP": "false",
            "SINGLE_USER_OWNER_ID": "default_user",
            "DEFAULT_MODEL": "test-model",
            "ALLOWED_MODELS": "test-model",
            "OPENAI_API_KEY": "",
            "OPENAI_BASE_URL": "",
            "SERPER_API_KEY": "",
        }
    )
    script = r'''
import Crawler
import Indexer
import Pagerank
import storage
import Searching
import llm_agent
import ai_search
import tools.config
import tools.security
import tools.privacy
import tools.ingestion
import tools.ingestion_models
import tools.rag
import tools.rag_tool
import tools.document_store
import tools.document_service
import tools.integrity
import tools.web_search
import tools.single_page
import tools.handbook
import tools.logger
import tools.job_store
import tools.rate_limit
import tools.bounded_pool
import tools.due_scheduler
import search_agent
import server

assert storage.StorageManager.__module__ == 'storage'
assert tools.rag.RAGLayer.__name__ == 'RAGLayer'
assert tools.document_store.DocumentStore.__name__ == 'DocumentStore'
assert tools.integrity.check_visual_entailment is not None
assert search_agent.SearchAgent.__name__ == 'SearchAgent'
assert server.app.version == '4.4.0'
server._cancel_scheduled_ingestions()
server._INGEST_EXECUTOR.shutdown(wait=False, cancel_futures=True)
server._QUERY_EXECUTOR.shutdown(wait=False, cancel_futures=True)
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr

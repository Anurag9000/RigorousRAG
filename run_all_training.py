#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,subprocess,sys
from pathlib import Path
REPOSITORY="Anurag9000/RigorousRAG";SHA="6043605115ddf934433380e892f1f238eb9e4af236c4063350f477bc5cb0d4dc";ROOT=Path(__file__).resolve().parent;CONTROLLER=ROOT/"tools"/"universal_training_controller.py"
PROFILE={"repository":REPOSITORY,"preferred_training_entrypoints":["train.py","training.py","run_training.py","scripts/train.py","scripts/train_all.py","scripts/run_training.py"],"preferred_dataset_entrypoints":["prepare_data.py","scripts/prepare_data.py","scripts/download_data.py","scripts/materialize_datasets.py","scripts/dataset_setup.py"],"dynamic_registry_covers":[],"extra_jobs":[],"ignore_entrypoints":["run_all_training.py"]}
def main():
 if not CONTROLLER.is_file() or hashlib.sha256(CONTROLLER.read_bytes()).hexdigest()!=SHA: raise RuntimeError("Pinned local training controller checksum mismatch")
 e=os.environ.copy();e["TRAINING_CONTROL_PROFILE"]=json.dumps(PROFILE,separators=(",",":"));e["TRAINING_CONTROL_REPO_ROOT"]=str(ROOT);return subprocess.call([sys.executable,str(CONTROLLER),*sys.argv[1:]],cwd=ROOT,env=e)
if __name__=="__main__":raise SystemExit(main())

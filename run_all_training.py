#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,subprocess,sys
from pathlib import Path
REPOSITORY="Anurag9000/RigorousRAG";SHA="0093db16bf5b13afcfbe0599484cac24a1e6c147";ROOT=Path(__file__).resolve().parent;CONTROLLER=ROOT/"tools"/"universal_training_controller_entry.py"
PROFILE={"repository":REPOSITORY,"preferred_training_entrypoints":["train.py","training.py","run_training.py","scripts/train.py","scripts/train_all.py","scripts/run_training.py"],"preferred_dataset_entrypoints":["prepare_data.py","scripts/prepare_data.py","scripts/download_data.py","scripts/materialize_datasets.py","scripts/dataset_setup.py"],"dynamic_registry_covers":[],"extra_jobs":[],"ignore_entrypoints":["run_all_training.py"],"strict_coverage":True,"require_native_resume":True,"require_exact_resume":True,"require_training_exact_resume":True,"require_training_early_stopping":True,"require_dag_enforcement":True,"require_model_surface_accounting":True,"require_literal_opf_mechanism_parity":True,"require_well_formed_training_exemptions":True}
def h(data):return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
def main():
 if not CONTROLLER.is_file() or h(CONTROLLER.read_bytes())!=SHA: raise RuntimeError("Pinned local training controller bootstrap checksum mismatch")
 e=os.environ.copy();e["TRAINING_CONTROL_PROFILE"]=json.dumps(PROFILE,separators=(",",":"));e["TRAINING_CONTROL_REPO_ROOT"]=str(ROOT);e.setdefault("TRAINING_CONTROL_TERMINATION_GRACE_SEC","30");return subprocess.call([sys.executable,str(CONTROLLER),*sys.argv[1:]],cwd=ROOT,env=e)
if __name__=="__main__":raise SystemExit(main())
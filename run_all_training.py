#!/usr/bin/env python3
from __future__ import annotations
import hashlib,os,subprocess,sys,urllib.request
from pathlib import Path
R="Anurag9000/RigorousRAG";B="dd3043cd8a36d50c03e5cfc27a17d58a74aa7bec";S="260337eaf24b70efe6331f71422b3f98715edcd3";AC="89a10cf30b4171aec27e67284df6900a6b1c68ca";AS="b34b4deca809f63a8f8db5d3189ea734f2c8c228";D=Path(__file__).resolve().parent;U=f"https://raw.githubusercontent.com/Anurag9000/RigorousRAG/{AC}/tools/repo_training_launcher_adapter.py"
def h(x):return hashlib.sha1(f"blob {len(x)}\0".encode()+x).hexdigest()
def main():
 p=D/".training_control"/"repo_training_launcher_adapter.py"
 if not p.is_file() or h(p.read_bytes())!=AS:
  p.parent.mkdir(parents=True,exist_ok=True);x=urllib.request.urlopen(U,timeout=60).read()
  if h(x)!=AS:raise RuntimeError("Pinned launcher adapter checksum mismatch")
  t=p.with_suffix(".tmp");t.write_bytes(x);os.replace(t,p)
 e=os.environ.copy();e["TRAINING_LAUNCHER_BASE_REPOSITORY"]=R;e["TRAINING_LAUNCHER_BASE_COMMIT"]=B;e["TRAINING_LAUNCHER_BASE_BLOB"]=S;e["TRAINING_CONTROL_REPO_ROOT"]=str(D);return subprocess.call([sys.executable,str(p),*sys.argv[1:]],cwd=D,env=e)
if __name__=="__main__":raise SystemExit(main())

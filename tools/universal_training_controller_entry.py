#!/usr/bin/env python3
from __future__ import annotations
import hashlib, os, subprocess, sys, urllib.request
from pathlib import Path

HOST_REPO="Anurag9000/RigorousRAG"
HOST_COMMIT="2439b2e77174164879cfcf5d8dec6e3caad3a19e"
FILES={
 "tools/universal_training_controller.py":"4353000e092ac286158c23500d91e898136fbab3",
 "tools/universal_training_controller_current.py":"04d5a3610faa84d1a8e951464cdd79df7d09f1b0",
}
def git_blob_sha(data: bytes)->str:
 return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
def main()->int:
 root=Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve()
 cache=root/".training_control"/"controller_host"/HOST_COMMIT
 for rel,expected in FILES.items():
  dst=cache/Path(rel).name
  valid=dst.is_file() and git_blob_sha(dst.read_bytes())==expected
  if not valid:
   url=f"https://raw.githubusercontent.com/{HOST_REPO}/{HOST_COMMIT}/{rel}"
   req=urllib.request.Request(url,headers={"User-Agent":"opf-training-controller-entry/1"})
   with urllib.request.urlopen(req,timeout=120) as r:data=r.read()
   actual=git_blob_sha(data)
   if actual!=expected:raise RuntimeError(f"controller blob mismatch {rel}: {actual} != {expected}")
   dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix(dst.suffix+".tmp");tmp.write_bytes(data);os.replace(tmp,dst)
 env=os.environ.copy();env["TRAINING_CONTROL_REPO_ROOT"]=str(root)
 return subprocess.call([sys.executable,str(cache/"universal_training_controller_current.py"),*sys.argv[1:]],cwd=root,env=env)
if __name__=="__main__":raise SystemExit(main())

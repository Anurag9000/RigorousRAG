#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,shutil,subprocess,sys,urllib.parse,urllib.request
from pathlib import Path

HOST_REPO="Anurag9000/RigorousRAG"
HOST_COMMIT="9c1d10852490715eb1c39c524ce19c9307c1b413"
FILES={
 "tools/universal_training_controller.py":"4353000e092ac286158c23500d91e898136fbab3",
 "tools/universal_training_controller_current.py":"09fe933dd520c7b97cdb0e86f5c5fbdc597336e4",
}
OPF_REPO="Anurag9000/OPF_ADP"
OPF_COMMIT="a34c31259bd5d5f58081e3766918f9df63017455"
OPF_FILES={
 "utils/opf_massive_suite_runner.py":"b97d47499c83bc6ed3a5753f7f3009b624c94868",
 "utils/runtime_tuning.py":"f1cbfc44e009701a5540a046f2cd6b9f41f16b74",
 "utils/ml_backends.py":"2fe2b24e530cab3d747c983c4457f4080703512f",
 "utils/logging_utils.py":"482ba94643aa921f49eebb835f29cf4930bb2498",
 "utils/opf_shared_defaults.py":"76ad434ecef1f708c835210d4bc86e0717999d99",
 "DNN/VANILLA/Dyn_DNN4OPF/utils/run_defaults.py":"dacb9a2c44d611c045fbb7512ba5327343f79a85",
}
INIT_FILES=("utils/__init__.py","DNN/__init__.py","DNN/VANILLA/__init__.py","DNN/VANILLA/Dyn_DNN4OPF/__init__.py","DNN/VANILLA/Dyn_DNN4OPF/utils/__init__.py")
def git_blob_sha(data:bytes)->str:return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
def atomic_write(path:Path,data:bytes)->None:
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_bytes(data);os.replace(tmp,path)
def fetch(url:str,headers:dict[str,str]|None=None)->bytes:
 req=urllib.request.Request(url,headers={"User-Agent":"opf-training-controller-entry/3",**(headers or {})})
 with urllib.request.urlopen(req,timeout=120) as r:return r.read()
def verified_local(root:Path,rel:str,expected:str)->bytes|None:
 roots=[]
 explicit=os.environ.get("OPF_REFERENCE_LOCAL_ROOT","").strip()
 if explicit:roots.append(Path(explicit).expanduser())
 roots.extend((root.parent/"OPF_ADP",Path.home()/"OPF_ADP",Path.home()/"projects"/"OPF_ADP",Path.home()/"Projects"/"OPF_ADP"))
 for base in roots:
  try:
   data=(base/rel).read_bytes()
  except Exception:continue
  if git_blob_sha(data)==expected:return data
 return None
def fetch_opf(root:Path,rel:str,expected:str)->bytes:
 data=verified_local(root,rel,expected)
 if data is not None:return data
 token=(os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "").strip()
 api=f"https://api.github.com/repos/{OPF_REPO}/contents/{urllib.parse.quote(rel,safe='/')}?ref={OPF_COMMIT}"
 if token:
  try:
   data=fetch(api,{"Accept":"application/vnd.github.raw+json","Authorization":f"Bearer {token}","X-GitHub-Api-Version":"2022-11-28"})
   if git_blob_sha(data)==expected:return data
  except Exception:pass
 gh=shutil.which("gh")
 if gh:
  try:
   data=subprocess.check_output([gh,"api","-H","Accept: application/vnd.github.raw+json",f"repos/{OPF_REPO}/contents/{rel}?ref={OPF_COMMIT}"],stderr=subprocess.DEVNULL)
   if git_blob_sha(data)==expected:return data
  except Exception:pass
 try:
  raw=f"https://raw.githubusercontent.com/{OPF_REPO}/{OPF_COMMIT}/{rel}"
  data=fetch(raw)
  if git_blob_sha(data)==expected:return data
 except Exception:pass
 raise RuntimeError(f"Cannot obtain verified private OPF reference file {rel}. Keep a sibling OPF_ADP checkout, set OPF_REFERENCE_LOCAL_ROOT, export GH_TOKEN/GITHUB_TOKEN, or authenticate the gh CLI.")
def prepare_opf_cache(root:Path)->None:
 cache=root/".training_control"/"opf_reference"/OPF_COMMIT;marker=cache/"REFERENCE.json"
 expected_marker={"repository":OPF_REPO,"commit":OPF_COMMIT,"files":OPF_FILES}
 try:
  valid=json.loads(marker.read_text(encoding="utf-8"))==expected_marker and all((cache/rel).is_file() and git_blob_sha((cache/rel).read_bytes())==sha for rel,sha in OPF_FILES.items())
 except Exception:valid=False
 if valid:return
 for rel,expected in OPF_FILES.items():
  data=fetch_opf(root,rel,expected)
  if git_blob_sha(data)!=expected:raise RuntimeError(f"Pinned OPF blob mismatch for {rel}")
  atomic_write(cache/rel,data)
 for rel in INIT_FILES:
  p=cache/rel
  if not p.exists():atomic_write(p,b"")
 atomic_write(marker,(json.dumps(expected_marker,indent=2,sort_keys=True)+"\n").encode())
def main()->int:
 root=Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve();cache=root/".training_control"/"controller_host"/HOST_COMMIT
 for rel,expected in FILES.items():
  dst=cache/Path(rel).name;valid=dst.is_file() and git_blob_sha(dst.read_bytes())==expected
  if not valid:
   data=fetch(f"https://raw.githubusercontent.com/{HOST_REPO}/{HOST_COMMIT}/{rel}")
   actual=git_blob_sha(data)
   if actual!=expected:raise RuntimeError(f"controller blob mismatch {rel}: {actual} != {expected}")
   atomic_write(dst,data)
 prepare_opf_cache(root)
 env=os.environ.copy();env["TRAINING_CONTROL_REPO_ROOT"]=str(root)
 return subprocess.call([sys.executable,str(cache/"universal_training_controller_current.py"),*sys.argv[1:]],cwd=root,env=env)
if __name__=="__main__":raise SystemExit(main())

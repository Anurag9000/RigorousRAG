#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,shutil,subprocess,sys,urllib.parse,urllib.request
from pathlib import Path

HOST_REPO="Anurag9000/RigorousRAG"
HOST_COMMIT="5592b48bafcac72fee8d0c9eeab7d00dedde1f0a"
FILES={
 "tools/universal_training_controller.py":"4353000e092ac286158c23500d91e898136fbab3",
 "tools/universal_training_controller_current.py":"09fe933dd520c7b97cdb0e86f5c5fbdc597336e4",
 "tools/universal_training_controller_dag.py":"3621f1fb0aeb843f1fb051cba074eedef67ac81e",
 "tools/universal_training_controller_exact_resume.py":"6f6311cbb7cdfa46d14ad5eb0adc8749c5080226",
 "tools/universal_training_controller_console.py":"89d4b8fde514ba426993d7068d3e4e6177600670",
 "tools/universal_training_controller_console_defaults.py":"a4aae98c861764e0ece3bdcfb87f72eb531f6381",
 "tools/universal_training_controller_subcommands.py":"969e2ea054c3d2f310d8ddd9a99e230e0f2cafd6",
 "tools/universal_training_controller_restart_exact.py":"6994858fc1294b79f0bb479afee7f88f453fb026",
 "tools/universal_training_controller_opf_grace.py":"73db03de6eeb6cdcca685e1ffbfde60f08969f1e",
 "tools/universal_training_controller_registry_scheduling.py":"d69ec5294a876c291f2ac6a210920e8e7233c510",
 "tools/universal_training_controller_training_contracts.py":"dec455d8fa2e2cc88113f4d382f072908cc9b1ac",
 "tools/universal_training_controller_profile_file.py":"43f7ef739ce92f94ea7e3c444d6b0a56c34f61e3",
 "tools/universal_training_controller_job_catalog_v2.py":"9e0643ae5075e0901ffe28f26024db7b28d37a34",
 "tools/universal_training_controller_large_catalog.py":"805fbe26d0b6e0251b11a808e629f96e1d210b16",
 "tools/universal_training_controller_opf_mechanism_audit.py":"1f19141e4ba6c15627125716139bb8872b453ec9",
 "tools/universal_training_controller_deferred.py":"72e33311f00d0d2353c671a4c1663b1e9d0daf6a",
 "tools/universal_training_controller_deferred_v2.py":"f0203b273ad58461178871a728c4ba18f73ab116",
 "tools/universal_training_controller_deferred_v3.py":"865378f887c269602676b1c7ca0859d25fd756b2",
 "tools/universal_training_controller_deferred_v4.py":"6dc85929f749cc1d5202d3481509e6db9b6aeb67",
 "tools/universal_training_controller_opf_reference_v2.py":"3b21bb8f60179e8c1e9b31d164ecf6799f9f9b5d",
 "tools/universal_training_controller_v20.py":"7ab4318b917fd3276249a02536ac02726ddc8eed",
}
OPF_REPO="Anurag9000/OPF_ADP"
OPF_COMMIT="2dfe664af88b95981da2b84b60f228a37156749f"
OPF_FILES={
 "utils/opf_massive_suite_runner.py":"b2ae3d04f9398df5c18c7c13f4c939bce46b930d",
 "utils/runtime_tuning.py":"f1cbfc44e009701a5540a046f2cd6b9f41f16b74",
 "utils/ml_backends.py":"c4cd5eaf783cd7ffbb92ab01ec743ef7cbd13d84",
 "utils/logging_utils.py":"482ba94643aa921f49eebb835f29cf4930bb2498",
 "utils/opf_shared_defaults.py":"bd76baa134b07567015d0151d5f14ba81dc667df",
 "DNN/VANILLA/Dyn_DNN4OPF/utils/run_defaults.py":"ff79e8c51f1fb21a11e4687989198ef0abb07491",
}
# Compatibility-only cache for repository-local binding scripts that have not yet
# been migrated. v20 never imports this scheduler.  It is now opt-in instead of
# being an unconditional clean-machine dependency.
LEGACY_OPF_COMMIT="a34c31259bd5d5f58081e3766918f9df63017455"
LEGACY_OPF_FILES={
 "utils/opf_massive_suite_runner.py":"b97d47499c83bc6ed3a5753f7f3009b624c94868",
 "utils/runtime_tuning.py":"f1cbfc44e009701a5540a046f2cd6b9f41f16b74",
 "utils/ml_backends.py":"2fe2b24e530cab3d747c983c4457f4080703512f",
 "utils/logging_utils.py":"482ba94643aa921f49eebb835f29cf4930bb2498",
 "utils/opf_shared_defaults.py":"76ad434ecef1f708c835210d4bc86e0717999d99",
 "DNN/VANILLA/Dyn_DNN4OPF/utils/run_defaults.py":"dacb9a2c44d611c045fbb7512ba5327343f79a85",
}
INIT_FILES=("utils/__init__.py","DNN/__init__.py","DNN/VANILLA/__init__.py","DNN/VANILLA/Dyn_DNN4OPF/__init__.py","DNN/VANILLA/Dyn_DNN4OPF/utils/__init__.py")
DIAGNOSTIC_FLAGS=frozenset({
 "--training-control-audit","--audit-training-coverage",
 "--list-training-jobs","--training-control-list-jobs",
 "--help","-h","--version",
})
def git_blob_sha(data:bytes)->str:return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
def atomic_write(path:Path,data:bytes)->None:
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_bytes(data);os.replace(tmp,path)
def fetch(url:str,headers:dict[str,str]|None=None)->bytes:
 req=urllib.request.Request(url,headers={"User-Agent":"opf-training-controller-entry/20",**(headers or {})})
 with urllib.request.urlopen(req,timeout=120) as r:return r.read()
def verified_local(root:Path,rel:str,expected:str)->bytes|None:
 roots=[]
 explicit=os.environ.get("OPF_REFERENCE_LOCAL_ROOT","").strip()
 if explicit:roots.append(Path(explicit).expanduser())
 roots.extend((root.parent/"OPF_ADP",Path.home()/"OPF_ADP",Path.home()/"projects"/"OPF_ADP",Path.home()/"Projects"/"OPF_ADP"))
 for base in roots:
  try:data=(base/rel).read_bytes()
  except Exception:continue
  if git_blob_sha(data)==expected:return data
 return None
def fetch_opf(root:Path,rel:str,expected:str,commit:str)->bytes:
 data=verified_local(root,rel,expected)
 if data is not None:return data
 token=(os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "").strip()
 api=f"https://api.github.com/repos/{OPF_REPO}/contents/{urllib.parse.quote(rel,safe='/')}?ref={commit}"
 if token:
  try:
   data=fetch(api,{"Accept":"application/vnd.github.raw+json","Authorization":f"Bearer {token}","X-GitHub-Api-Version":"2022-11-28"})
   if git_blob_sha(data)==expected:return data
  except Exception:pass
 gh=shutil.which("gh")
 if gh:
  try:
   data=subprocess.check_output([gh,"api","-H","Accept: application/vnd.github.raw+json",f"repos/{OPF_REPO}/contents/{rel}?ref={commit}"],stderr=subprocess.DEVNULL)
   if git_blob_sha(data)==expected:return data
  except Exception:pass
 try:
  raw=f"https://raw.githubusercontent.com/{OPF_REPO}/{commit}/{rel}";data=fetch(raw)
  if git_blob_sha(data)==expected:return data
 except Exception:pass
 raise RuntimeError(f"Cannot obtain verified private OPF reference file {rel}@{commit}. Keep a sibling OPF_ADP checkout, set OPF_REFERENCE_LOCAL_ROOT, export a cross-repository GH_TOKEN/GITHUB_TOKEN, or authenticate the gh CLI.")
def prepare_reference_cache(root:Path,commit:str,files:dict[str,str])->None:
 cache=root/".training_control"/"opf_reference"/commit;marker=cache/"REFERENCE.json";expected_marker={"repository":OPF_REPO,"commit":commit,"files":files}
 try:valid=json.loads(marker.read_text(encoding="utf-8"))==expected_marker and all((cache/rel).is_file() and git_blob_sha((cache/rel).read_bytes())==sha for rel,sha in files.items())
 except Exception:valid=False
 if valid:return
 for rel,expected in files.items():
  data=fetch_opf(root,rel,expected,commit)
  if git_blob_sha(data)!=expected:raise RuntimeError(f"Pinned OPF blob mismatch for {rel}@{commit}")
  atomic_write(cache/rel,data)
 for rel in INIT_FILES:
  p=cache/rel
  if not p.exists():atomic_write(p,b"")
 atomic_write(marker,(json.dumps(expected_marker,indent=2,sort_keys=True)+"\n").encode())
def diagnostic_only(argv:list[str])->bool:
 """Return True only for modes that are contractually non-executing.

 Audits/list/help must remain usable on a clean checkout with no OPF credentials.
 A mixed invocation is treated as executing unless every option that can request
 work is absent; this fails closed rather than accidentally bypassing runtime
 materialization for a real training run.
 """
 if not argv:return False
 return any(arg in DIAGNOSTIC_FLAGS for arg in argv)
def main()->int:
 root=Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve();cache=root/".training_control"/"controller_host"/HOST_COMMIT
 for rel,expected in FILES.items():
  dst=cache/Path(rel).name;valid=dst.is_file() and git_blob_sha(dst.read_bytes())==expected
  if not valid:
   data=fetch(f"https://raw.githubusercontent.com/{HOST_REPO}/{HOST_COMMIT}/{rel}");actual=git_blob_sha(data)
   if actual!=expected:raise RuntimeError(f"controller blob mismatch {rel}: {actual} != {expected}")
   atomic_write(dst,data)
 if not diagnostic_only(list(sys.argv[1:])):
  prepare_reference_cache(root,OPF_COMMIT,OPF_FILES)
  if os.environ.get("TRAINING_CONTROL_PREPARE_LEGACY_OPF","").strip().lower() in {"1","true","yes","on"}:
   prepare_reference_cache(root,LEGACY_OPF_COMMIT,LEGACY_OPF_FILES)
 env=os.environ.copy();env["TRAINING_CONTROL_REPO_ROOT"]=str(root)
 return subprocess.call([sys.executable,str(cache/"universal_training_controller_v20.py"),*sys.argv[1:]],cwd=root,env=env)
if __name__=="__main__":raise SystemExit(main())

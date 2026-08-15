"""Declarative experiment DAGs compiled into deterministic execution tasks."""
from __future__ import annotations
import hashlib,json
from dataclasses import asdict,dataclass,field
from typing import Any,Mapping,Sequence
def _text(v:Any,l:str,m:int=500,empty:bool=False)->str:
    if not isinstance(v,str):raise ValueError(f"{l} must be string")
    s=" ".join(v.replace("\x00"," ").split())
    if (not s and not empty) or len(s)>m:raise ValueError(f"{l} is invalid")
    return s
def _canon(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
@dataclass(frozen=True)
class ExperimentStage:
    stage_id:str; kind:str; config:Mapping[str,Any]; depends_on:tuple[str,...]=(); resources:Mapping[str,float]=field(default_factory=dict); seed:int|None=None
    def __post_init__(self):
        object.__setattr__(self,"stage_id",_text(self.stage_id,"stage_id",256)); object.__setattr__(self,"kind",_text(self.kind,"kind",64).lower()); object.__setattr__(self,"depends_on",tuple(dict.fromkeys(_text(x,"dependency",256) for x in self.depends_on)))
        if not isinstance(self.config,Mapping) or len(self.config)>256:raise ValueError("config invalid")
        _canon(dict(self.config))
        if not isinstance(self.resources,Mapping) or len(self.resources)>64:raise ValueError("resources invalid")
        clean={}
        for k,v in self.resources.items():
            n=_text(str(k),"resource",100); f=float(v)
            if f<0:raise ValueError("resource values must be non-negative")
            clean[n]=f
        object.__setattr__(self,"resources",clean)
        if self.seed is not None and (isinstance(self.seed,bool) or not isinstance(self.seed,int)):raise ValueError("seed invalid")
@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id:str; stages:tuple[ExperimentStage,...]; metadata:Mapping[str,str]=field(default_factory=dict)
    def __post_init__(self):
        object.__setattr__(self,"experiment_id",_text(self.experiment_id,"experiment_id",256)); ids={s.stage_id for s in self.stages}
        if not self.stages or len(ids)!=len(self.stages) or len(self.stages)>10000:raise ValueError("stages invalid")
        if any(not set(s.depends_on).issubset(ids-{s.stage_id}) for s in self.stages):raise ValueError("unknown/self dependency")
        indeg={i:0 for i in ids}; children={i:[] for i in ids}
        for s in self.stages:
            indeg[s.stage_id]=len(s.depends_on)
            for d in s.depends_on:children[d].append(s.stage_id)
        q=sorted(i for i,d in indeg.items() if d==0); visited=[]
        while q:
            n=q.pop(0);visited.append(n)
            for c in sorted(children[n]):
                indeg[c]-=1
                if indeg[c]==0:q.append(c);q.sort()
        if len(visited)!=len(ids):raise ValueError("experiment DAG contains cycle")
    @property
    def fingerprint(self)->str:return hashlib.sha256(_canon(asdict(self))).hexdigest()
@dataclass(frozen=True)
class ExperimentTask:
    task_id:str; experiment_id:str; stage_id:str; stage_sha256:str; dependency_task_ids:tuple[str,...]; config:Mapping[str,Any]; resources:Mapping[str,float]; seed:int|None

def compile_experiment(spec:ExperimentSpec)->tuple[ExperimentTask,...]:
    by={s.stage_id:s for s in spec.stages}; tasks={}; remaining=set(by)
    while remaining:
        progressed=False
        for sid in sorted(tuple(remaining)):
            stage=by[sid]
            if all(d in tasks for d in stage.depends_on):
                stage_sha=hashlib.sha256(_canon(asdict(stage))).hexdigest(); tid=hashlib.sha256(_canon((spec.experiment_id,sid,stage_sha))).hexdigest()
                tasks[sid]=ExperimentTask(tid,spec.experiment_id,sid,stage_sha,tuple(tasks[d].task_id for d in stage.depends_on),dict(stage.config),dict(stage.resources),stage.seed);remaining.remove(sid);progressed=True
        if not progressed:raise RuntimeError("experiment compilation stalled")
    return tuple(tasks[s.stage_id] for s in spec.stages)
__all__=["ExperimentSpec","ExperimentStage","ExperimentTask","compile_experiment"]

"""Directed basin/reach topology reasoning for hydrology evidence retrieval."""
from __future__ import annotations
import hashlib,json
from collections import defaultdict,deque
from dataclasses import asdict,dataclass
from typing import Any,Mapping,Sequence
from tools.hydrology_domain import GeoPoint
def _text(v:Any,l:str,m:int=500)->str:
    if not isinstance(v,str) or not v.strip() or len(v.strip())>m or "\x00" in v:raise ValueError(f"{l} invalid")
    return " ".join(v.split())
def _canon(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False,default=str).encode()
@dataclass(frozen=True)
class HydroNode:
    node_id:str; kind:str; location:GeoPoint|None=None; source_id:str=""
    def __post_init__(self):
        object.__setattr__(self,"node_id",_text(self.node_id,"node_id",256));k=_text(self.kind,"kind",64).lower()
        if k not in {"junction","reservoir","dam","gauge","boundary","source","sink","other"}:raise ValueError("unsupported hydro node kind")
        object.__setattr__(self,"kind",k)
        if self.location is not None and not isinstance(self.location,GeoPoint):raise ValueError("location invalid")
        object.__setattr__(self,"source_id",_text(self.source_id,"source_id",1000) if self.source_id else "")
@dataclass(frozen=True)
class HydroReach:
    reach_id:str; upstream_node_id:str; downstream_node_id:str; length_m:float; source_id:str=""; attributes:Mapping[str,str]=None
    def __post_init__(self):
        for n in ("reach_id","upstream_node_id","downstream_node_id"):object.__setattr__(self,n,_text(getattr(self,n),n,256))
        if self.upstream_node_id==self.downstream_node_id or float(self.length_m)<0:raise ValueError("reach invalid")
        object.__setattr__(self,"length_m",float(self.length_m));object.__setattr__(self,"source_id",_text(self.source_id,"source_id",1000) if self.source_id else "")
        object.__setattr__(self,"attributes",dict(self.attributes or {}))
@dataclass(frozen=True)
class HydroPath:
    node_ids:tuple[str,...]; reach_ids:tuple[str,...]; length_m:float
class HydroNetwork:
    def __init__(self,nodes:Sequence[HydroNode],reaches:Sequence[HydroReach]):
        self.nodes={n.node_id:n for n in nodes};self.reaches={r.reach_id:r for r in reaches}
        if len(self.nodes)!=len(nodes) or len(self.reaches)!=len(reaches):raise ValueError("duplicate hydro IDs")
        self.down=defaultdict(list);self.up=defaultdict(list)
        indeg={n:0 for n in self.nodes}
        for r in reaches:
            if r.upstream_node_id not in self.nodes or r.downstream_node_id not in self.nodes:raise ValueError("reach references unknown node")
            self.down[r.upstream_node_id].append(r);self.up[r.downstream_node_id].append(r);indeg[r.downstream_node_id]+=1
        q=deque(n for n,d in indeg.items() if d==0);visited=0
        while q:
            n=q.popleft();visited+=1
            for r in self.down[n]:
                indeg[r.downstream_node_id]-=1
                if indeg[r.downstream_node_id]==0:q.append(r.downstream_node_id)
        if visited!=len(self.nodes):raise ValueError("hydrologic network contains a directed cycle")
    def upstream_nodes(self,node_id:str,*,max_hops:int=100)->tuple[str,...]:
        target=_text(node_id,"node_id",256);seen={target};q=deque([(target,0)]);out=[]
        while q:
            current,depth=q.popleft()
            if depth>=max_hops:continue
            for r in sorted(self.up[current],key=lambda x:x.reach_id):
                n=r.upstream_node_id
                if n not in seen:seen.add(n);out.append(n);q.append((n,depth+1))
        return tuple(out)
    def downstream_nodes(self,node_id:str,*,max_hops:int=100)->tuple[str,...]:
        target=_text(node_id,"node_id",256);seen={target};q=deque([(target,0)]);out=[]
        while q:
            current,depth=q.popleft()
            if depth>=max_hops:continue
            for r in sorted(self.down[current],key=lambda x:x.reach_id):
                n=r.downstream_node_id
                if n not in seen:seen.add(n);out.append(n);q.append((n,depth+1))
        return tuple(out)
    def paths(self,source_id:str,target_id:str,*,max_paths:int=1000,max_hops:int=100)->tuple[HydroPath,...]:
        source=_text(source_id,"source_id",256);target=_text(target_id,"target_id",256);out=[];q=deque([(source,(source,),(),0.0)])
        while q and len(out)<max_paths:
            current,nodes,reaches,length=q.popleft()
            if current==target:out.append(HydroPath(nodes,reaches,length));continue
            if len(reaches)>=max_hops:continue
            for r in sorted(self.down[current],key=lambda x:x.reach_id):
                if r.downstream_node_id in nodes:continue
                q.append((r.downstream_node_id,(*nodes,r.downstream_node_id),(*reaches,r.reach_id),length+r.length_m))
        return tuple(out)
    @property
    def fingerprint(self)->str:return hashlib.sha256(_canon({"nodes":[asdict(self.nodes[k]) for k in sorted(self.nodes)],"reaches":[asdict(self.reaches[k]) for k in sorted(self.reaches)]})).hexdigest()
__all__=["HydroNetwork","HydroNode","HydroPath","HydroReach"]

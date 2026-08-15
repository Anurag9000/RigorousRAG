"""Explicit versioned schema migrations with content fingerprints and no implicit coercion."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from typing import Any,Callable,Mapping,Sequence
_VERSION=re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
def _text(v:Any,l:str,m:int=256)->str:
    if not isinstance(v,str) or not v.strip() or len(v.strip())>m or "\x00" in v:raise ValueError(f"{l} invalid")
    return v.strip()
def _canon(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def _version(v:str)->str:
    s=_text(v,"version",32)
    if not _VERSION.fullmatch(s):raise ValueError("version must be semantic x.y.z")
    return s
MigrationFn=Callable[[Mapping[str,Any]],Mapping[str,Any]]
@dataclass(frozen=True)
class SchemaMigration:
    schema_id:str; from_version:str; to_version:str; migration_id:str; transform:MigrationFn
    def __post_init__(self):
        object.__setattr__(self,"schema_id",_text(self.schema_id,"schema_id"));object.__setattr__(self,"from_version",_version(self.from_version));object.__setattr__(self,"to_version",_version(self.to_version));object.__setattr__(self,"migration_id",_text(self.migration_id,"migration_id"))
        if self.from_version==self.to_version or not callable(self.transform):raise ValueError("migration invalid")
class SchemaEvolutionRegistry:
    def __init__(self):self._migrations={}
    def register(self,migration:SchemaMigration)->None:
        key=(migration.schema_id,migration.from_version,migration.to_version)
        prior=self._migrations.get(key)
        if prior is not None and prior.migration_id!=migration.migration_id:raise ValueError("migration path collision")
        self._migrations[key]=migration
    def path(self,schema_id:str,from_version:str,to_version:str)->tuple[SchemaMigration,...]:
        sid=_text(schema_id,"schema_id");source=_version(from_version);target=_version(to_version)
        if source==target:return ()
        frontier=[(source,())];seen={source}
        while frontier:
            current,path=frontier.pop(0)
            candidates=sorted((m for (s,f,t),m in self._migrations.items() if s==sid and f==current),key=lambda m:m.to_version)
            for migration in candidates:
                if migration.to_version==target:return (*path,migration)
                if migration.to_version not in seen:seen.add(migration.to_version);frontier.append((migration.to_version,(*path,migration)))
        raise KeyError(f"no schema migration path {sid} {source}->{target}")
    def migrate(self,schema_id:str,payload:Mapping[str,Any],from_version:str,to_version:str)->tuple[Mapping[str,Any],str]:
        if not isinstance(payload,Mapping):raise ValueError("payload must be mapping")
        current=dict(payload);history=[]
        for migration in self.path(schema_id,from_version,to_version):
            result=migration.transform(current)
            if not isinstance(result,Mapping):raise RuntimeError("schema migration did not return mapping")
            current=dict(result);_canon(current);history.append(migration.migration_id)
        digest=hashlib.sha256(_canon({"schema":schema_id,"from":from_version,"to":to_version,"history":history,"payload":current})).hexdigest()
        return current,digest
__all__=["SchemaEvolutionRegistry","SchemaMigration"]

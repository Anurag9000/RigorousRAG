"""Owner-scoped project collaboration ACLs with explicit least-privilege roles."""
from __future__ import annotations
import hashlib,json,time
from dataclasses import asdict,dataclass
from typing import Any,Mapping,Sequence
from tools.security import normalize_owner_id
_ROLES={"viewer":frozenset({"project.read","session.read","report.read"}),"reviewer":frozenset({"project.read","session.read","report.read","claim.review"}),"editor":frozenset({"project.read","project.write","session.read","session.write","report.read","report.write","claim.review"}),"owner":frozenset({"project.read","project.write","session.read","session.write","report.read","report.write","claim.review","acl.manage","project.delete"})}
def _text(v:Any,l:str,m:int=500)->str:
    if not isinstance(v,str) or not v.strip() or len(v.strip())>m or "\x00" in v: raise ValueError(f"{l} is invalid")
    return " ".join(v.split())
def _canon(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
@dataclass(frozen=True)
class ProjectGrant:
    project_id:str; principal_id:str; role:str; granted_by:str; granted_at:float; expires_at:float|None=None
    def __post_init__(self):
        object.__setattr__(self,"project_id",_text(self.project_id,"project_id",256)); object.__setattr__(self,"principal_id",normalize_owner_id(self.principal_id)); object.__setattr__(self,"granted_by",normalize_owner_id(self.granted_by)); r=_text(self.role,"role",32).lower()
        if r not in _ROLES: raise ValueError("unsupported ACL role")
        object.__setattr__(self,"role",r)
        if self.expires_at is not None and float(self.expires_at)<=float(self.granted_at): raise ValueError("ACL expiration must follow grant time")
    @property
    def fingerprint(self)->str:return hashlib.sha256(_canon(asdict(self))).hexdigest()
class ProjectACL:
    def __init__(self,*,project_id:str,owner_id:str):
        self.project_id=_text(project_id,"project_id",256); self.owner_id=normalize_owner_id(owner_id); self._grants={self.owner_id:ProjectGrant(self.project_id,self.owner_id,"owner",self.owner_id,time.time())}
    def grant(self,*,actor_id:str,principal_id:str,role:str,expires_at:float|None=None)->ProjectGrant:
        self.require(actor_id,"acl.manage"); principal=normalize_owner_id(principal_id)
        if principal==self.owner_id and role!="owner": raise ValueError("project owner role may not be downgraded")
        grant=ProjectGrant(self.project_id,principal,role,normalize_owner_id(actor_id),time.time(),expires_at); self._grants[principal]=grant; return grant
    def revoke(self,*,actor_id:str,principal_id:str)->None:
        self.require(actor_id,"acl.manage"); principal=normalize_owner_id(principal_id)
        if principal==self.owner_id: raise ValueError("project owner grant may not be revoked")
        self._grants.pop(principal,None)
    def permissions(self,principal_id:str,*,now:float|None=None)->frozenset[str]:
        principal=normalize_owner_id(principal_id); grant=self._grants.get(principal)
        if grant is None:return frozenset()
        current=time.time() if now is None else float(now)
        if grant.expires_at is not None and current>=grant.expires_at:return frozenset()
        return _ROLES[grant.role]
    def require(self,principal_id:str,permission:str)->None:
        selected=_text(permission,"permission",100)
        if selected not in self.permissions(principal_id): raise PermissionError("project permission is not granted")
    def grants(self)->tuple[ProjectGrant,...]:return tuple(sorted(self._grants.values(),key=lambda item:item.principal_id))
    @property
    def fingerprint(self)->str:return hashlib.sha256(_canon([asdict(item) for item in self.grants()])).hexdigest()
__all__=["ProjectACL","ProjectGrant"]

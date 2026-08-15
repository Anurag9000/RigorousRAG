"""Citation-linked numerical consistency checks for structured answer quantities."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Mapping,Sequence
from tools.numerical_reasoning import Quantity,UnitRegistry,default_unit_registry
@dataclass(frozen=True)
class QuantitativeAssertion:
    assertion_id:str; value:float; unit:str; citation_ids:tuple[str,...]; tolerance_abs:float=0.0; tolerance_rel:float=1e-6
    def __post_init__(self):
        if not isinstance(self.assertion_id,str) or not self.assertion_id.strip():raise ValueError("assertion_id invalid")
        if isinstance(self.value,bool) or not math.isfinite(float(self.value)):raise ValueError("value invalid")
        if not isinstance(self.unit,str) or not self.unit.strip():raise ValueError("unit invalid")
        if not self.citation_ids:raise ValueError("quantitative assertion requires citations")
        if self.tolerance_abs<0 or self.tolerance_rel<0:raise ValueError("tolerances must be non-negative")
@dataclass(frozen=True)
class QuantitativeEvidence:
    evidence_id:str; quantity:Quantity
@dataclass(frozen=True)
class NumericalCheck:
    assertion_id:str; status:str; matched_evidence_ids:tuple[str,...]; observed_delta:float|None; message:str

def check_assertions(assertions:Sequence[QuantitativeAssertion],evidence:Mapping[str,QuantitativeEvidence],*,registry:UnitRegistry|None=None)->tuple[NumericalCheck,...]:
    units=registry or default_unit_registry();results=[]
    for assertion in assertions:
        matches=[];best=None;conversion_errors=0
        for citation_id in assertion.citation_ids:
            item=evidence.get(citation_id)
            if item is None:continue
            try: converted=item.quantity.convert(assertion.unit,units)
            except (KeyError,ValueError):conversion_errors+=1;continue
            delta=abs(converted.value-float(assertion.value));threshold=max(assertion.tolerance_abs,assertion.tolerance_rel*max(abs(float(assertion.value)),abs(converted.value),1.0))
            if best is None or delta<best:best=delta
            if delta<=threshold:matches.append(item.evidence_id)
        if matches:results.append(NumericalCheck(assertion.assertion_id,"consistent",tuple(dict.fromkeys(matches)),best,"At least one cited quantitative evidence item matches within tolerance."))
        elif best is not None:results.append(NumericalCheck(assertion.assertion_id,"inconsistent",(),best,"Cited quantitative evidence does not match the asserted value within tolerance."))
        elif conversion_errors:results.append(NumericalCheck(assertion.assertion_id,"dimension_mismatch",(),None,"Cited quantities could not be converted to the asserted unit/dimension."))
        else:results.append(NumericalCheck(assertion.assertion_id,"missing_evidence",(),None,"No cited structured quantitative evidence was available."))
    return tuple(results)
__all__=["NumericalCheck","QuantitativeAssertion","QuantitativeEvidence","check_assertions"]

from tools.adapter_registry import AdapterRegistry, AdapterVersion
from tools.hf_late_interaction_backend import HuggingFaceLateInteractionBackend
from tools.hf_multimodal_backend import HuggingFaceMultimodalBackend
from tools.hf_sparse_backend import HuggingFaceSparseBackend
from tools.retrieval_artifact_binding import active_retrieval_spec
from tools.retrieval_model_contracts import MultimodalInput, RetrievalModelSpec, finite_vector


def make_spec(mode):
    return RetrievalModelSpec(mode, "model/name", "revision-1", "a" * 64)


def test_contracts_validate_mode_and_vector():
    assert finite_vector([1, 2.5]) == (1.0, 2.5)
    assert MultimodalInput(text="question").text == "question"
    try:
        make_spec("bad")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid mode accepted")


def test_active_artifact_binding_pins_checksum():
    registry = AdapterRegistry()
    registry.register(AdapterVersion(name="splade", version="1.0.0", kind="retrieval-sparse",
                                     artifact_uri="model://splade", checksum_sha256="b" * 64))
    registry.promote("splade", "1.0.0")
    value = active_retrieval_spec(registry, "splade", "sparse", "model/name", "rev-1")
    assert value.mode == "sparse" and value.checksum_sha256 == "b" * 64


def test_optional_backends_reject_wrong_mode_before_importing_heavy_dependencies():
    for factory, value in (
        (HuggingFaceSparseBackend, make_spec("late-interaction")),
        (HuggingFaceLateInteractionBackend, make_spec("sparse")),
    ):
        try:
            factory(value)
        except ValueError:
            pass
        else:
            raise AssertionError("backend accepted wrong mode")
    try:
        HuggingFaceMultimodalBackend(make_spec("sparse"), image_decoder=lambda value: value)
    except ValueError:
        pass
    else:
        raise AssertionError("multimodal backend accepted wrong mode")

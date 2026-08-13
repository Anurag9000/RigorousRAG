from evaluation.accelerator_probe import AcceleratorSample


class TorchCudaSampleReader:
    def __init__(self, device: int = 0) -> None:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch is unavailable") from exc
        self.torch = torch
        self.device = device

    def __call__(self) -> AcceleratorSample:
        cuda = self.torch.cuda
        if not cuda.is_available() or self.device >= cuda.device_count():
            return AcceleratorSample("torch-cuda", "unavailable")
        cuda.synchronize(self.device)
        return AcceleratorSample(
            backend="torch-cuda",
            device=str(cuda.get_device_name(self.device)),
            allocated_bytes=int(cuda.memory_allocated(self.device)),
            peak_bytes=int(cuda.max_memory_allocated(self.device)),
        )


__all__ = ["TorchCudaSampleReader"]

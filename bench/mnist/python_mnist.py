#!/usr/bin/env python3
"""Optional PyTorch counterpart of examples/showcase.s (no torchvision)."""

import argparse
import json
import math
import struct
import time
from pathlib import Path

import torch
from torch import nn


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def seeded_normal(shape, seed, device):
    # Sputnik uses an independent CPU generator per layer, then moves to device.
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(device)


class Conv2d(nn.Module):
    def __init__(self, incoming, outgoing, kernel, *, stride=1, padding=0, seed=0, device="cpu"):
        super().__init__()
        self.weight = nn.Parameter(seeded_normal((outgoing, incoming, kernel, kernel), seed, device)
                                   / math.sqrt(incoming * kernel * kernel))
        self.stride, self.padding = (stride, stride), (padding, padding)

    def forward(self, x):
        return nn.functional.conv2d(x, self.weight, None, self.stride, self.padding)


class Linear(nn.Module):
    def __init__(self, incoming, outgoing, *, seed=0, device="cpu"):
        super().__init__()
        self.weight = nn.Parameter(seeded_normal((outgoing, incoming), seed, device) * 0.1)
        self.bias = nn.Parameter(torch.zeros(outgoing, device=device))

    def forward(self, x):
        # Same matmul + add as Sputnik; nn.Linear normally uses fused addmm.
        return x.matmul(self.weight.transpose(0, 1)) + self.bias


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, *, stride=1, seed=0, device="cpu"):
        super().__init__()
        self.body = nn.Sequential(
            Conv2d(incoming, outgoing, 3, stride=stride, padding=1, seed=seed, device=device),
            nn.BatchNorm2d(outgoing, device=device), nn.ReLU(),
            Conv2d(outgoing, outgoing, 3, padding=1, seed=seed + 1, device=device),
            nn.BatchNorm2d(outgoing, device=device),
        )
        self.shortcut = nn.Identity() if incoming == outgoing and stride == 1 else nn.Sequential(
            Conv2d(incoming, outgoing, 1, stride=stride, seed=seed + 2, device=device),
            nn.BatchNorm2d(outgoing, device=device),
        )

    def forward(self, x):
        return (self.body(x) + self.shortcut(x)).relu()


class SmallResNet(nn.Module):
    def __init__(self, *, seed=7, device="cpu"):
        super().__init__()
        self.features = nn.Sequential(
            Conv2d(1, 16, 3, padding=1, seed=seed, device=device),
            nn.BatchNorm2d(16, device=device), nn.ReLU(),
            ResidualBlock(16, 16, seed=seed + 1, device=device),
            ResidualBlock(16, 32, stride=2, seed=seed + 4, device=device),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.classifier = Linear(32, 10, seed=seed + 7, device=device)

    def forward(self, x):
        return self.classifier(self.features(x))


class MatchedAdamW:
    """Scalar reference for the Sputnik AdamW formula before tensor-list packing."""

    def __init__(self, parameters, *, lr=0.003, weight_decay=0.0001):
        self.parameters = list(parameters)
        self.lr, self.weight_decay = lr, weight_decay
        self.first = [torch.zeros_like(p) for p in self.parameters]
        self.second = [torch.zeros_like(p) for p in self.parameters]
        self.steps = [0] * len(self.parameters)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None

    @torch.no_grad()
    def step(self):
        for index, p in enumerate(self.parameters):
            gradient = p.grad
            if gradient is None:
                continue
            p.mul_(1 - self.lr * self.weight_decay)
            self.steps[index] += 1
            first = self.first[index] * 0.9 + gradient * (1 - 0.9)
            second = self.second[index] * 0.999 + gradient.square() * (1 - 0.999)
            self.first[index], self.second[index] = first, second
            corrected_first = first / (1 - 0.9 ** self.steps[index])
            corrected_second = second / (1 - 0.999 ** self.steps[index])
            p.sub_(corrected_first / (corrected_second.sqrt() + 1e-8) * self.lr)


class PackedAdamW(MatchedAdamW):
    """MNIST reference with the same flat moments and list updates as Sputnik."""

    def __init__(self, parameters):
        super().__init__(parameters)
        self.sizes = [p.numel() for p in self.parameters]
        self.shapes = [p.shape for p in self.parameters]
        self.first = torch.cat([value.reshape(-1) for value in self.first])
        self.second = torch.cat([value.reshape(-1) for value in self.second])
        self.steps = 0

    @torch.no_grad()
    def step(self):
        gradients = [p.grad for p in self.parameters]
        if any(value is None for value in gradients):
            raise RuntimeError("packed MNIST reference requires every parameter gradient")
        torch._foreach_mul_(self.parameters, 1 - self.lr * self.weight_decay)
        gradient = torch.cat([value.reshape(-1) for value in gradients])
        self.steps += 1
        self.first = self.first * 0.9 + gradient * (1 - 0.9)
        self.second = self.second * 0.999 + gradient.square() * (1 - 0.999)
        corrected_first = self.first / (1 - 0.9 ** self.steps)
        corrected_second = self.second / (1 - 0.999 ** self.steps)
        update = corrected_first / (corrected_second.sqrt() + 1e-8) * self.lr
        pieces = [part.reshape(shape) for part, shape in zip(update.split(self.sizes), self.shapes)]
        torch._foreach_sub_(self.parameters, pieces)


def read_idx(path):
    data = Path(path).read_bytes()
    if len(data) > 64 * 1024 * 1024:
        raise ValueError(f"IDX file exceeds 64 MiB: {path}")
    if len(data) < 4 or data[:3] != b"\x00\x00\x08":
        raise ValueError(f"expected uint8 IDX data: {path}")
    rank = data[3]
    offset = 4 + rank * 4
    if not 1 <= rank <= 8 or len(data) < offset:
        raise ValueError(f"invalid IDX dimensions: {path}")
    shape = struct.unpack(f">{rank}I", data[4:offset])
    if not all(shape) or math.prod(shape) != len(data) - offset:
        raise ValueError(f"IDX payload size mismatch: {path}")
    return torch.frombuffer(bytearray(data[offset:]), dtype=torch.uint8).reshape(shape)


def load_mnist(directory, split, limit):
    prefix = "train" if split == "train" else "t10k"
    images = read_idx(directory / f"{prefix}-images-idx3-ubyte")
    labels = read_idx(directory / f"{prefix}-labels-idx1-ubyte")
    if labels.ndim != 1 or list(images.shape) != [labels.numel(), 28, 28]:
        raise ValueError("MNIST image and label dimensions disagree")
    if limit is not None:
        if not 1 <= limit <= labels.numel():
            raise ValueError("MNIST limit must be within the split")
        images, labels = images.narrow(0, 0, limit), labels.narrow(0, 0, limit)
    return ((images.unsqueeze(1).to(dtype=torch.float32) / 255.0 - 0.1307) / 0.3081,
            labels.to(dtype=torch.int64))


class Batches:
    """TensorDataset fast path with Sputnik's epoch seed and index_select batches."""

    def __init__(self, dataset, batch_size, *, shuffle=False, seed=0):
        self.dataset, self.batch_size = dataset, batch_size
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def __iter__(self):
        count = self.dataset[1].numel()
        order = (torch.randperm(count, generator=torch.Generator().manual_seed(self.seed + self.epoch))
                 .tolist() if self.shuffle else list(range(count)))
        for start in range(0, count, self.batch_size):
            indices = torch.tensor(order[start:start + self.batch_size], dtype=torch.int64)
            yield tuple(t.index_select(0, indices.to(t.device)) for t in self.dataset)


class BatchProfile:
    def __init__(self, device):
        self.device = device
        self.seconds = dict.fromkeys(
            ("batch", "transfer", "zero_grad", "forward", "backward", "optimizer", "metrics"), 0.0)

    def start(self):
        synchronize(self.device)
        self.started = time.perf_counter()

    def mark(self, key):
        synchronize(self.device)
        now = time.perf_counter()
        self.seconds[key] += now - self.started
        self.started = now


def train_epoch(model, batches, loss_fn, optimizer, device, profile=None):
    model.train()
    total_loss, seen = 0.0, 0
    if profile:
        profile.start()
    for features, labels in batches:
        if profile:
            profile.mark("batch")
        features, labels = features.to(device), labels.to(device)
        if profile:
            profile.mark("transfer")
        optimizer.zero_grad()
        if profile:
            profile.mark("zero_grad")
        loss = loss_fn(model(features), labels)
        if profile:
            profile.mark("forward")
        loss.backward()
        if profile:
            profile.mark("backward")
        optimizer.step()
        if profile:
            profile.mark("optimizer")
        total_loss += loss.item() * labels.numel()
        seen += labels.numel()
        if profile:
            profile.mark("metrics")
    return total_loss / seen


@torch.no_grad()
def evaluate(model, batches, loss_fn, device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for features, labels in batches:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        total_loss += loss_fn(logits, labels).item() * labels.numel()
        correct += logits.argmax(dim=1).eq(labels).sum().item()
        seen += labels.numel()
    return total_loss / seen, correct / seen


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("build/datasets/mnist"))
    parser.add_argument("--checkpoint", type=Path, default=Path("build/mnist-python.pt"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--optimizer", choices=("matched", "packed", "torch"), default="matched")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch size must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; run with Metal access or select --device cpu")
    # Sputnik's native batch_norm explicitly passes cudnn_enabled=false.
    torch.backends.cudnn.enabled = False
    phase = time.perf_counter()
    train = load_mnist(args.data, "train", args.train_limit)
    test = load_mnist(args.data, "test", args.test_limit)
    data_seconds = time.perf_counter() - phase
    phase = time.perf_counter()
    batches = Batches(train, args.batch_size, shuffle=True, seed=args.seed)
    test_batches = Batches(test, args.batch_size)
    model = SmallResNet(seed=args.seed, device=args.device)
    loss_fn = nn.CrossEntropyLoss()
    if args.optimizer == "torch":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001,
                                     foreach=False, fused=False)
    else:
        optimizer_type = PackedAdamW if args.optimizer == "packed" else MatchedAdamW
        optimizer = optimizer_type(model.parameters())
    parameters = sum(p.numel() for p in model.parameters())
    synchronize(args.device)
    setup_seconds = time.perf_counter() - phase
    epochs = []
    for epoch in range(args.epochs):
        batches.epoch = epoch
        profile = BatchProfile(args.device) if args.profile else None
        synchronize(args.device)
        phase = time.perf_counter()
        train_loss = train_epoch(model, batches, loss_fn, optimizer, args.device, profile)
        synchronize(args.device)
        train_seconds = time.perf_counter() - phase
        phase = time.perf_counter()
        test_loss, accuracy = evaluate(model, test_batches, loss_fn, args.device)
        synchronize(args.device)
        eval_seconds = time.perf_counter() - phase
        epochs.append(dict(epoch=epoch + 1, train_loss=train_loss, test_loss=test_loss,
                           accuracy=accuracy, train_seconds=train_seconds, eval_seconds=eval_seconds,
                           profile=profile.seconds if profile else None))
        print(f"epoch {epoch + 1}/{args.epochs} | train {train_loss} | test {test_loss} | accuracy {accuracy * 100}%",
              flush=True)
        lr = 0.0003 + (0.003 - 0.0003) * (1 + math.cos(math.pi * (epoch + 1) / args.epochs)) / 2
        if args.optimizer != "torch":
            optimizer.lr = lr
        else:
            optimizer.param_groups[0]["lr"] = lr
    phase = time.perf_counter()
    # Sputnik state_dict snapshots also clone parameters and buffers before save.
    torch.save({key: value.detach().clone() for key, value in model.state_dict().items()}, args.checkpoint)
    synchronize(args.device)
    save_seconds = time.perf_counter() - phase
    phase = time.perf_counter()
    restored = SmallResNet(seed=args.seed, device=args.device)
    restored.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    synchronize(args.device)
    restore_seconds = time.perf_counter() - phase
    phase = time.perf_counter()
    loss, accuracy = evaluate(restored, test_batches, loss_fn, args.device)
    synchronize(args.device)
    restore_eval_seconds = time.perf_counter() - phase
    if abs(loss - epochs[-1]["test_loss"]) > 1e-7 or accuracy != epochs[-1]["accuracy"]:
        raise RuntimeError("checkpoint changed test metrics")
    result = dict(schema="sputniktorch.mnist.sample.v1", device=args.device, version=torch.__version__,
                  threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                  train_count=train[1].numel(), test_count=test[1].numel(), parameters=parameters,
                  batch_size=args.batch_size, seed=args.seed, optimizer=args.optimizer, profiled=args.profile,
                  data_seconds=data_seconds, setup_seconds=setup_seconds, epochs=epochs,
                  save_seconds=save_seconds, restore_seconds=restore_seconds,
                  restore_eval_seconds=restore_eval_seconds, checkpoint_loss=loss,
                  checkpoint_accuracy=accuracy, workload_seconds=time.perf_counter() - started)
    print("MNIST_BENCH " + json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

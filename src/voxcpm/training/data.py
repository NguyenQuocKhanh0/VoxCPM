import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random
import argbind
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset
import torchaudio
from ..model.voxcpm import VoxCPMConfig
from ..modules.audiovae import AudioVAE
from .packers import AudioFeatureProcessingPacker


DEFAULT_TEXT_COLUMN = "text"
DEFAULT_AUDIO_COLUMN = "audio"
DEFAULT_ID_COLUMN = "dataset_id"


@argbind.bind()
def load_audio_text_datasets(
    train_manifest: str,
    val_manifest: str = "",
    text_column: str = DEFAULT_TEXT_COLUMN,
    audio_column: str = DEFAULT_AUDIO_COLUMN,
    dataset_id_column: str = DEFAULT_ID_COLUMN,
    sample_rate: int = 16_000,
    num_proc: int = 1,
) -> Tuple[Dataset, Optional[Dataset]]:
    data_files = {"train": train_manifest}
    if val_manifest:
        data_files["validation"] = val_manifest

    dataset_dict: DatasetDict = load_dataset("json", data_files=data_files)

    def prepare(ds: Dataset) -> Dataset:
        if audio_column not in ds.column_names:
            raise ValueError(f"Expected '{audio_column}' column in manifest.")
        # We cast to Audio to ensure proper handling during training, 
        # but for length calculation we might need raw path or duration if available.
        # HF datasets usually don't compute duration automatically for 'Audio' column.
        ds = ds.cast_column(audio_column, Audio(sampling_rate=sample_rate))
        if audio_column != DEFAULT_AUDIO_COLUMN:
            ds = ds.rename_column(audio_column, DEFAULT_AUDIO_COLUMN)
        if text_column != DEFAULT_TEXT_COLUMN:
            ds = ds.rename_column(text_column, DEFAULT_TEXT_COLUMN)
        if dataset_id_column and dataset_id_column in ds.column_names:
            if dataset_id_column != DEFAULT_ID_COLUMN:
                ds = ds.rename_column(dataset_id_column, DEFAULT_ID_COLUMN)
        else:
            ds = ds.add_column(DEFAULT_ID_COLUMN, [0] * len(ds))
        return ds

    train_ds = prepare(dataset_dict["train"])
    val_ds = prepare(dataset_dict["validation"]) if "validation" in dataset_dict else None
    return train_ds, val_ds


def compute_sample_lengths(
    ds: Dataset,
    audio_vae_fps: int = 25,
    patch_size: int = 1,
) -> List[int]:
    """
    预估每个样本经过 packer 之后的大致序列长度（text+audio），用于过滤超长样本。

    逻辑与 AudioFeatureProcessingPacker / AudioVAE 一致：
    - 文本长度: len(text_ids)
    - 音频长度:
        duration(s) * audio_vae_fps -> 近似 VAE 帧数 t_vae
        t_seq = ceil(t_vae / patch_size)
    - 序列总长约为: text_len + t_seq + 2
    """
    lengths: List[int] = []

    has_duration = "duration" in ds.column_names

    for i in range(len(ds)):
        item = ds[i]
        text_len = len(item["text_ids"])

        # 音频时长（尽量不解码；若 manifest 里已有 duration 列则优先使用）
        if has_duration:
            duration = float(item["duration"])
        else:
            audio = item[DEFAULT_AUDIO_COLUMN]
            duration = len(audio["array"]) / float(audio["sampling_rate"])

        t_vae = math.ceil(duration * audio_vae_fps)
        t_seq = math.ceil(t_vae / patch_size)

        total_len = text_len + t_seq + 2
        lengths.append(total_len)

    return lengths


class HFVoxCPMDataset(TorchDataset):
    """
    Thin wrapper around a tokenized HuggingFace dataset that returns
    PyTorch-friendly samples.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        audio = item[DEFAULT_AUDIO_COLUMN]
        return {
            "text_ids": item["text_ids"],
            "audio_array": audio["array"],
            "audio_sampling_rate": audio["sampling_rate"],
            "dataset_id": item.get(DEFAULT_ID_COLUMN, 0),
            "is_prompt": item.get("is_prompt", False),
        }

    @staticmethod
    def pad_sequences(seqs: List[torch.Tensor], pad_value: float):
        if not seqs:
            return torch.empty(0)
        max_len = max(seq.shape[0] for seq in seqs)
        padded = []
        for seq in seqs:
            if seq.shape[0] < max_len:
                pad_width = (0, max_len - seq.shape[0])
                seq = torch.nn.functional.pad(seq, pad_width, value=pad_value)
            padded.append(seq)
        return torch.stack(padded)

    @classmethod
    def collate_fn(cls, batch: List[Dict]):
        text_tensors = [torch.tensor(sample["text_ids"], dtype=torch.int32) for sample in batch]
        audio_tensors = [torch.tensor(sample["audio_array"], dtype=torch.float32) for sample in batch]
        dataset_ids = torch.tensor([sample["dataset_id"] for sample in batch], dtype=torch.int32)
        is_prompts = [bool(sample.get("is_prompt", False)) for sample in batch]

        text_padded = cls.pad_sequences(text_tensors, pad_value=-100)
        audio_padded = cls.pad_sequences(audio_tensors, pad_value=-100.0)
        task_ids = torch.ones(text_padded.size(0), dtype=torch.int32)

        return {
            "text_tokens": text_padded,
            "audio_tokens": audio_padded,
            "task_ids": task_ids,
            "dataset_ids": dataset_ids,
            "is_prompts": is_prompts,
        }


class BatchProcessor:
    """
    Wraps ``AudioFeatureProcessingPacker`` so the training loop can mirror
    the minicpm-audio mechanics.
    """

    def __init__(
        self,
        *,
        config: VoxCPMConfig,
        audio_vae: AudioVAE,
        dataset_cnt: int,
        device: torch.device,
        duration_control: Optional[dict] = None,
        sample_rate: Optional[int] = None,
    ):
        self.device = device
        self.dataset_cnt = dataset_cnt
        self.audio_vae = audio_vae
        self.audio_vae.to(device)
        self.sample_rate = int(sample_rate) if sample_rate is not None else int(getattr(audio_vae, "sample_rate", 44100))
        self.duration_control = duration_control or {}
        self.dur_enabled = bool(self.duration_control.get("enabled", False))
        self.tempo_aug = self.duration_control.get("tempo_augment", {"enabled": False})
        self.packer = AudioFeatureProcessingPacker(
            dataset_cnt=dataset_cnt,
            max_len=config.max_length,
            patch_size=config.patch_size,
            feat_dim=config.feat_dim,
            audio_vae=self.audio_vae,
        )

    def _maybe_tempo_augment(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Tempo augment WITHOUT pitch shift, using sox 'tempo'.
        Expect wav shape [B, T] or [T]. Returns same dtype/device as input.
        """
        cfg = self.tempo_aug
        if not cfg or not bool(cfg.get("enabled", False)):
            return wav
        p = float(cfg.get("p", 0.7))
        if random.random() >= p:
            return wav
        fmin, fmax = float(cfg.get("min", 0.85)), float(cfg.get("max", 1.15))
        factor = random.uniform(fmin, fmax)

        # torchaudio sox expects [channels, time]
        if wav.dim() == 1:
            wav_ = wav.unsqueeze(0)
        elif wav.dim() == 2:
            # assume [B, T] -> treat as mono per sample by looping
            # to keep batch semantics stable, apply per-sample
            outs = []
            for i in range(wav.size(0)):
                x = wav[i].unsqueeze(0)
                y, _ = torchaudio.sox_effects.apply_effects_tensor(x.cpu(), sr, effects=[["tempo", f"{factor}"]])
                outs.append(y.squeeze(0))
            return torch.nn.utils.rnn.pad_sequence(outs, batch_first=True).to(wav.device, wav.dtype)
        else:
            return wav

        y, _ = torchaudio.sox_effects.apply_effects_tensor(wav_.cpu(), sr, effects=[["tempo", f"{factor}"]])
        return y.squeeze(0).to(wav.device, wav.dtype) 

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # audio_tokens here are raw waveform padded by collate_fn
        audio_tokens = batch["audio_tokens"].to(self.device)
        text_tokens = batch["text_tokens"].to(self.device)
        task_ids = batch["task_ids"].to(self.device)
        dataset_ids = batch["dataset_ids"].to(self.device)
        # tempo augment BEFORE packer -> BEFORE audio_vae.encode
        # Note: padding value from collate_fn is -100.0; packer should ignore via its own logic.
        # If your packer treats -100 as valid samples, replace padding to 0 here.
        if self.dur_enabled and self.tempo_aug.get("enabled", False):
            # replace pad with 0 to avoid sox artifacts
            audio_tokens = torch.where(audio_tokens == -100.0, torch.zeros_like(audio_tokens), audio_tokens)
            audio_tokens = self._maybe_tempo_augment(audio_tokens, self.sample_rate)
        packed = self.packer(
            audio_tokens=audio_tokens,
            text_tokens=text_tokens,
            task_ids=task_ids,
            dataset_ids=dataset_ids,
            is_prompts=batch["is_prompts"],
        )
        # duration_patches supervision:
        # loss_mask is typically 1 for audio steps (patch tokens), 0 else.
        # Use it as ground-truth target length for duration control.
        if self.dur_enabled and "loss_mask" in packed:
            # shape: [B]
            packed["duration_patches"] = packed["loss_mask"].sum(dim=1).to(torch.long)
        return packed


def build_dataloader(
    hf_dataset: Dataset,
    *,
    accelerator,
    batch_size: int,
    num_workers: int,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    torch_dataset = HFVoxCPMDataset(hf_dataset)
    # Standard padding-based batching; Accelerator will attach DistributedSampler if needed.
    return accelerator.prepare_dataloader(
        torch_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=HFVoxCPMDataset.collate_fn,
        drop_last=drop_last,
    )


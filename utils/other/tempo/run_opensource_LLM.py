#!/usr/bin/env python3
"""
run_opensource_LLM.py

Run Tencent's open-source HY-MT1.5 translation model (1.8B recommended for laptops)
with either:
  - Transformers (PyTorch; uses Apple Metal/MPS on Mac if available)
  - llama.cpp via llama-cpp-python (recommended on Mac for small-footprint GGUF quantizations)

This script is intentionally VERY verbose (prints lots of diagnostics) to help debugging.

NOTE ON LICENSE (IMPORTANT):
Tencent's HY-MT1.5 models are released under the "Tencent HY Community License Agreement".
The license text states it "does not apply in the European Union, United Kingdom and South Korea"
and defines "Territory" as worldwide excluding those regions.
You are responsible for ensuring your usage complies with the license and applicable law.

References (for your convenience):
- HF model card: https://huggingface.co/tencent/HY-MT1.5-1.8B
- HF GGUF model card: https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF
- License: https://huggingface.co/tencent/HY-MT1.5-1.8B/blob/main/License.txt
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import os
import platform
import sys
import textwrap
import time
import warnings
from typing import Any, Dict, List, Optional


# =============================================================================
# Logging / timing helpers
# =============================================================================

def hr(char: str = "=", width: int = 88) -> str:
    return char * width

def section(title: str) -> None:
    print("\n" + hr("="))
    print(title)
    print(hr("="))

def ts() -> str:
    # ISO-like local timestamp (no timezone info)
    return time.strftime("%Y-%m-%d %H:%M:%S")

def fmt_dur(seconds: float) -> str:
    ms = seconds * 1000.0
    if ms < 1000.0:
        return f"{ms:.3f} ms"
    return f"{seconds:.3f} s"

def kv(key: str, value: Any, indent: int = 0) -> None:
    pad = " " * indent
    print(f"{pad}{key}: {value}")

def info(msg: str) -> None:
    print(f"{ts()} [INFO] {msg}")

def warn(msg: str) -> None:
    print(f"{ts()} [WARN] {msg}")

def err(msg: str) -> None:
    print(f"{ts()} [ERROR] {msg}")

class StepTimer:
    """Context manager that prints start/end + elapsed time."""
    def __init__(self, label: str, also_print_end: bool = True) -> None:
        self.label = label
        self.also_print_end = also_print_end
        self.t0 = 0.0

    def __enter__(self):
        info(f"START: {self.label}")
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        if exc_type is None:
            if self.also_print_end:
                info(f"END:   {self.label}  (elapsed {fmt_dur(dt)})")
        else:
            err(f"FAILED: {self.label}  (after {fmt_dur(dt)}) — {exc_type.__name__}: {exc}")
        # Do not suppress exceptions
        return False

def human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "unknown"
    if n < 0:
        return str(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    i = 0
    while v >= 1024.0 and i < len(units) - 1:
        v /= 1024.0
        i += 1
    return f"{v:.2f} {units[i]}"

def try_import(name: str):
    try:
        mod = __import__(name)
        return mod
    except Exception:
        return None

def print_mps_memory(torch_mod) -> None:
    # Best-effort; not all torch builds expose these
    try:
        if hasattr(torch_mod, "mps") and hasattr(torch_mod.mps, "current_allocated_memory"):
            allocated = torch_mod.mps.current_allocated_memory()
            reserved = getattr(torch_mod.mps, "current_reserved_memory", lambda: None)()
            kv("mps_current_allocated", human_bytes(int(allocated)))
            kv("mps_current_reserved", human_bytes(int(reserved)) if reserved is not None else "unknown")
    except Exception as e:
        warn(f"Could not query MPS memory: {e}")

def print_system_diagnostics() -> None:
    section("Environment diagnostics")
    kv("python", sys.version.replace("\n", " "))
    kv("executable", sys.executable)
    kv("platform", platform.platform())
    kv("machine", platform.machine())
    kv("processor", platform.processor())
    kv("cwd", os.getcwd())
    kv("pid", os.getpid())
    kv("hf_home", os.environ.get("HF_HOME", "(default)"))
    kv("hf_cache", os.environ.get("HUGGINGFACE_HUB_CACHE", "(default)"))
    kv("transformers_cache", os.environ.get("TRANSFORMERS_CACHE", "(default)"))
    kv("torch_mps_fallback", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "(not set)"))
    kv("pythonpath", os.environ.get("PYTHONPATH", "(not set)"))

    # Optional: process RSS memory (if psutil installed)
    psutil = try_import("psutil")
    if psutil is not None:
        try:
            proc = psutil.Process(os.getpid())
            rss = proc.memory_info().rss
            kv("process_rss", human_bytes(int(rss)))
        except Exception as e:
            warn(f"psutil present but could not read memory info: {e}")
    else:
        kv("process_rss", "(psutil not installed)")

def install_hints() -> None:
    print("\nInstall hints:")
    print("  pip install -U pip")
    print("  # Transformers backend:")
    print("  pip install torch 'transformers==4.56.0'")
    print("  # llama.cpp backend on Apple Silicon (Metal wheels):")
    print("  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal")
    print("  pip install huggingface_hub")


# =============================================================================
# Prompt templates (from model card)
# =============================================================================

def prompt_basic_en(target_language: str, source_text: str) -> str:
    return f"Translate the following segment into {target_language}, without additional explanation.\n\n{source_text}"

def prompt_basic_zh(target_language: str, source_text: str) -> str:
    return f"将以下文本翻译为{target_language}，注意只需要输出翻译后的结果，不要额外解释：\n\n{source_text}"

def prompt_terminology(target_language: str, source_text: str, source_term: str, target_term: str) -> str:
    return (
        f"参考下面的翻译：\n{source_term} 翻译成 {target_term}\n\n"
        f"将以下文本翻译为{target_language}，注意只需要输出翻译后的结果，不要额外解释：\n{source_text}"
    )

def prompt_contextual(target_language: str, source_text: str, context: str) -> str:
    context = context.strip()
    return (
        f"{context}\n"
        f"参考上面的信息，把下面的文本翻译成{target_language}，注意不需要翻译上文，也不要额外解释：\n"
        f"{source_text}"
    )

def prompt_formatted_to_zh(src_text_with_format: str) -> str:
    return (
        "将以下<source></source>之间的文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释，"
        "原文中的<sn></sn>标签表示标签内文本包含格式信息，需要在译文中相应的位置尽量保留该标签。"
        "输出格式为：<target>str</target>\n\n"
        f"<source>{src_text_with_format}</source>"
    )


# =============================================================================
# Test cases (mental health + biomedical content)
# =============================================================================

@dataclasses.dataclass
class TestCase:
    name: str
    target_language: str
    kind: str  # basic | terminology | context | formatted
    source_text: str
    source_term: Optional[str] = None
    target_term: Optional[str] = None
    context: Optional[str] = None

def build_default_test_suite() -> List[TestCase]:
    return [
        TestCase(
            name="EN→ES clinical intake + biomarkers (formatting, abbreviations, scales)",
            target_language="Spanish",
            kind="basic",
            source_text=textwrap.dedent("""\
                Clinical intake summary (de-identified):

                - Presenting problems: recurrent major depressive disorder (DSM-5), comorbid generalized anxiety disorder,
                  chronic insomnia with sleep-onset and maintenance difficulties.
                - Severity: PHQ-9 = 18 (moderately severe); GAD-7 = 15 (severe).
                - Current medications: sertraline 100 mg qAM; melatonin 3 mg HS PRN.
                - Risk: passive suicidal ideation reported intermittently; denies plan/intent; no access to lethal means.
                - Plan:
                  1) Initiate CBT-I protocol; sleep diary for 14 days.
                  2) Psychoeducation: SSRI onset latency (2–6 weeks) and common side effects.
                  3) Safety plan: crisis hotline + local emergency number; identify protective factors and supports.
                  4) Follow-up in 2 weeks to reassess symptoms and adherence.

                Research note:
                The patient is enrolled in an fMRI study assessing default mode network (DMN) connectivity as a biomarker
                for treatment response; primary outcomes include changes in DMN coupling and symptom trajectories.
            """).strip(),
        ),
        TestCase(
            name="EN→ZH terminology intervention (force key clinical terms)",
            target_language="中文",
            kind="terminology",
            source_term="major depressive disorder",
            target_term="重度抑郁障碍",
            source_text=textwrap.dedent("""\
                The study targets major depressive disorder and examines whether baseline resting-state fMRI measures
                (e.g., default mode network connectivity) predict response to SSRI treatment. Secondary endpoints include
                anxiety symptom reduction and improvements in insomnia after CBT-I.
            """).strip(),
        ),
        TestCase(
            name="ES→EN context-aware translation (keep names/pronouns consistent, preserve nuance)",
            target_language="English",
            kind="context",
            context=textwrap.dedent("""\
                Context (do NOT translate this context, only use it to resolve ambiguity):
                - Speaker A is the clinician; Speaker B is the patient.
                - The patient previously said they stopped sertraline because of nausea but wants to try again.
                - The clinician uses respectful, non-judgmental language and avoids alarmist phrasing.
            """).strip(),
            source_text="Sí, lo dejé por las náuseas, pero quiero intentarlo otra vez si podemos empezar más despacio.",
        ),
        TestCase(
            name="Formatted translation into Chinese (preserve <sn> tags and structure)",
            target_language="中文",
            kind="formatted",
            source_text=textwrap.dedent("""\
                <sn>PHQ-9</sn>: 18
                <sn>GAD-7</sn>: 15
                Medication: <sn>sertraline</sn> 100 mg qAM
                Plan: CBT-I + follow-up in 2 weeks
            """).strip(),
        ),
    ]


# =============================================================================
# Prompt builder
# =============================================================================

def build_prompt(
    kind: str,
    prompt_style: str,
    target_language: str,
    source_text: str,
    source_term: Optional[str] = None,
    target_term: Optional[str] = None,
    context: Optional[str] = None
) -> str:
    kind = kind.lower()
    prompt_style = prompt_style.lower()

    if kind == "formatted":
        return prompt_formatted_to_zh(source_text)

    if kind == "terminology":
        if not source_term or not target_term:
            raise ValueError("terminology prompt requires --source_term and --target_term")
        return prompt_terminology(target_language, source_text, source_term, target_term)

    if kind == "context":
        if not context:
            raise ValueError("context prompt requires --context or --context_file")
        return prompt_contextual(target_language, source_text, context)

    # basic
    if prompt_style == "en":
        return prompt_basic_en(target_language, source_text)
    if prompt_style == "zh":
        return prompt_basic_zh(target_language, source_text)

    # auto: use zh-style if target looks like Chinese
    tl = target_language.strip().lower()
    if tl in {"zh", "chinese", "中文", "simplified chinese", "traditional chinese", "zh-hant"}:
        return prompt_basic_zh(target_language, source_text)
    return prompt_basic_en(target_language, source_text)


# =============================================================================
# Backend: Transformers (PyTorch)
# =============================================================================

class TransformersRunner:
    def __init__(
        self,
        model_id: str,
        device_preference: str,
        dtype_choice: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        warmup: bool,
    ) -> None:
        self.model_id = model_id
        self.device_preference = device_preference
        self.dtype_choice = dtype_choice
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.warmup = warmup

        self.torch = None
        self.transformers = None
        self.tokenizer = None
        self.model = None
        self.device = None
        self.dtype = None

    def _pick_device(self):
        torch = self.torch
        assert torch is not None

        pref = (self.device_preference or "").lower()
        if pref in {"cpu", "cuda", "mps"}:
            if pref == "cuda" and not torch.cuda.is_available():
                warn("device_preference=cuda but torch.cuda.is_available() is False; falling back.")
            elif pref == "mps" and not torch.backends.mps.is_available():
                warn("device_preference=mps but torch.backends.mps.is_available() is False; falling back.")
            else:
                return torch.device(pref)

        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _pick_dtype(self):
        torch = self.torch
        assert torch is not None

        choice = (self.dtype_choice or "").lower()
        if choice in {"auto", ""}:
            if self.device.type == "cuda":
                return torch.bfloat16
            if self.device.type == "mps":
                return torch.float16
            return torch.float32

        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if choice not in mapping:
            raise ValueError(f"Unsupported dtype choice: {self.dtype_choice} (use auto/fp16/bf16/fp32)")
        return mapping[choice]

    def _configure_transformers_logging(self):
        tfm = self.transformers
        if tfm is None:
            return
        try:
            # Ensure we see informative logs + progress bars
            tfm.logging.set_verbosity_info()
            tfm.logging.enable_default_handler()
            tfm.logging.enable_explicit_format()
            # progress bar is enabled by default, but keep explicit
            try:
                tfm.utils.logging.enable_progress_bar()
            except Exception:
                pass
        except Exception as e:
            warn(f"Could not configure transformers logging: {e}")

    def load(self) -> None:
        section("Backend = transformers (PyTorch) — loading model (ultra-verbose)")

        # Make warnings more visible
        def _warn_fmt(message, category, filename, lineno, line=None):
            return f"{ts()} [PYWARN] {category.__name__}: {message}  ({filename}:{lineno})\n"
        warnings.formatwarning = _warn_fmt  # type: ignore

        with StepTimer("Import torch"):
            try:
                import torch  # type: ignore
            except Exception as e:
                err(f"Failed to import torch: {e}")
                install_hints()
                raise

        with StepTimer("Import transformers"):
            try:
                import transformers  # type: ignore
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            except Exception as e:
                err(f"Failed to import transformers: {e}")
                install_hints()
                raise

        self.torch = torch
        self.transformers = transformers
        self._configure_transformers_logging()

        # Versions & environment
        section("Library versions")
        kv("torch_version", torch.__version__)
        kv("transformers_version", getattr(transformers, "__version__", "unknown"))
        hub = try_import("huggingface_hub")
        if hub is not None:
            kv("huggingface_hub_version", getattr(hub, "__version__", "unknown"))
        acc = try_import("accelerate")
        if acc is not None:
            kv("accelerate_version", getattr(acc, "__version__", "unknown"))
        st = try_import("safetensors")
        if st is not None:
            kv("safetensors_version", getattr(st, "__version__", "unknown"))

        # Choose device/dtype
        self.device = self._pick_device()
        self.dtype = self._pick_dtype()

        section("Device / dtype selection")
        kv("model_id", self.model_id)
        kv("device_preference", self.device_preference)
        kv("chosen_device", str(self.device))
        kv("chosen_dtype", str(self.dtype))
        kv("cuda_available", torch.cuda.is_available())
        kv("mps_available", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        try:
            kv("mps_built", getattr(torch.backends.mps, "is_built", lambda: "unknown")())
        except Exception:
            kv("mps_built", "unknown")

        if self.device.type == "cuda":
            kv("cuda_device_name", torch.cuda.get_device_name(0))
            kv("cuda_total_mem", human_bytes(torch.cuda.get_device_properties(0).total_memory))
        if self.device.type == "mps":
            info("MPS selected (Apple Metal). If you see unsupported ops, consider setting:")
            info("  export PYTORCH_ENABLE_MPS_FALLBACK=1")
            print_mps_memory(torch)

        # Load tokenizer
        with StepTimer("AutoTokenizer.from_pretrained(...)"):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        section("Tokenizer details")
        try:
            kv("tokenizer_class", self.tokenizer.__class__.__name__)
            kv("model_max_length", getattr(self.tokenizer, "model_max_length", "unknown"))
            kv("padding_side", getattr(self.tokenizer, "padding_side", "unknown"))
            kv("truncation_side", getattr(self.tokenizer, "truncation_side", "unknown"))
            kv("bos_token", getattr(self.tokenizer, "bos_token", None))
            kv("eos_token", getattr(self.tokenizer, "eos_token", None))
            kv("pad_token", getattr(self.tokenizer, "pad_token", None))
            kv("unk_token", getattr(self.tokenizer, "unk_token", None))
        except Exception as e:
            warn(f"Could not print tokenizer details: {e}")

        # Load model (use dtype parameter if supported, else torch_dtype)
        from transformers import AutoModelForCausalLM  # type: ignore
        fp_sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
        supports_dtype_kw = "dtype" in fp_sig.parameters
        supports_torch_dtype_kw = "torch_dtype" in fp_sig.parameters

        section("Model download / initialization")
        kv("from_pretrained_supports_dtype_kw", supports_dtype_kw)
        kv("from_pretrained_supports_torch_dtype_kw", supports_torch_dtype_kw)

        kwargs: Dict[str, Any] = dict(
            low_cpu_mem_usage=True,
        )
        if supports_dtype_kw:
            kwargs["dtype"] = self.dtype
        elif supports_torch_dtype_kw:
            kwargs["torch_dtype"] = self.dtype
        else:
            warn("Neither `dtype` nor `torch_dtype` is in from_pretrained signature; not passing dtype explicitly.")

        with StepTimer("AutoModelForCausalLM.from_pretrained(...)"):
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

        # Print basic model info before moving
        section("Model object (pre-device move)")
        try:
            kv("model_class", self.model.__class__.__name__)
            cfg = getattr(self.model, "config", None)
            kv("has_config", cfg is not None)
            if cfg is not None:
                kv("config_architectures", getattr(cfg, "architectures", None))
                kv("config_model_type", getattr(cfg, "model_type", None))
                kv("config_vocab_size", getattr(cfg, "vocab_size", None))
                kv("config_max_position_embeddings", getattr(cfg, "max_position_embeddings", None))
        except Exception as e:
            warn(f"Could not print model info: {e}")

        # Count parameters (can be a bit slow but useful)
        with StepTimer("Count model parameters"):
            try:
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                kv("total_params", f"{total_params:,}")
                kv("trainable_params", f"{trainable_params:,}")
            except Exception as e:
                warn(f"Could not count parameters: {e}")

        # Move model to device and eval
        with StepTimer("Move model to device (.to)"):
            self.model.to(self.device)

        if self.device.type == "mps":
            section("MPS memory after model.to(device)")
            print_mps_memory(torch)

        with StepTimer("model.eval()"):
            self.model.eval()

        info("Model loaded successfully.")

        # Optional warmup: helps detect runtime issues early
        if self.warmup:
            section("Warmup (tiny generation to validate end-to-end)")
            warm_prompt = "Translate the following segment into Spanish, without additional explanation.\n\nHello!"
            try:
                _ = self.generate(warm_prompt, print_prompt=True, print_output=False)
                info("Warmup complete.")
            except Exception as e:
                err(f"Warmup failed: {e}")
                raise

    def generate(self, prompt: str, print_prompt: bool = True, print_output: bool = True) -> Dict[str, Any]:
        if self.model is None or self.tokenizer is None or self.torch is None or self.device is None:
            raise RuntimeError("TransformersRunner not loaded. Call load() first.")

        torch = self.torch
        tokenizer = self.tokenizer
        model = self.model
        device = self.device

        section("Transformers inference (ultra-verbose)")

        kv("prompt_chars", len(prompt))
        if print_prompt:
            print("\n--- Prompt (verbatim) ---")
            print(prompt)
            print("--- End prompt ---\n")

        # Prepare messages (as per HF model card example)
        messages = [{"role": "user", "content": prompt}]

        with StepTimer("tokenizer.apply_chat_template(..., tokenize=True)"):
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            )

        try:
            kv("input_ids_dtype", str(input_ids.dtype))
        except Exception:
            pass
        kv("input_ids_shape", tuple(input_ids.shape))
        kv("input_token_count", int(input_ids.shape[-1]))

        # Show a small preview of token ids (useful for debugging templates)
        try:
            ids_list = input_ids[0].tolist()
            preview_n = min(16, len(ids_list))
            kv("input_token_id_preview", ids_list[:preview_n])
            kv("input_token_id_tail", ids_list[-preview_n:] if len(ids_list) > preview_n else ids_list)
        except Exception as e:
            warn(f"Could not preview token ids: {e}")

        with StepTimer("Move input_ids to device"):
            input_ids = input_ids.to(device)

        if device.type == "mps":
            section("MPS memory before generate")
            print_mps_memory(torch)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
        )

        section("Generation parameters")
        for k, v in gen_kwargs.items():
            kv(k, v)

        with StepTimer("model.generate(...)"):
            with torch.inference_mode():
                out = model.generate(input_ids, **gen_kwargs)

        # Output stats
        total_tokens = int(out.shape[-1])
        gen_tokens = total_tokens - int(input_ids.shape[-1])
        kv("total_output_tokens", total_tokens)
        kv("generated_tokens", gen_tokens)

        # Decode only the generated continuation
        with StepTimer("Decode generated tokens"):
            gen_ids = out[0, input_ids.shape[-1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        if device.type == "mps":
            section("MPS memory after generate")
            print_mps_memory(torch)

        if print_output:
            print("\n--- Model output ---")
            print(text)
            print("--- End output ---\n")

        return {
            "text": text,
            "generated_tokens": gen_tokens,
        }


# =============================================================================
# Backend: llama.cpp (GGUF) via llama-cpp-python
# =============================================================================

class LlamaCppRunner:
    def __init__(
        self,
        repo_id: str,
        gguf_filename: str,
        n_ctx: int,
        n_threads: int,
        n_gpu_layers: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> None:
        self.repo_id = repo_id
        self.gguf_filename = gguf_filename
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        self.llama = None
        self.model_path = None

    def load(self) -> None:
        section("Backend = llama.cpp (llama-cpp-python) — loading GGUF model (ultra-verbose)")
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except Exception as e:
            err(f"Failed to import huggingface_hub: {e}")
            install_hints()
            raise

        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            err(f"Failed to import llama_cpp (llama-cpp-python): {e}")
            install_hints()
            raise

        section("GGUF selection")
        kv("hf_repo_id", self.repo_id)
        kv("gguf_filename", self.gguf_filename)
        kv("n_ctx", self.n_ctx)
        kv("n_threads", self.n_threads)
        kv("n_gpu_layers", self.n_gpu_layers)

        with StepTimer("hf_hub_download(GGUF)"):
            self.model_path = hf_hub_download(repo_id=self.repo_id, filename=self.gguf_filename)

        kv("local_model_path", self.model_path)

        with StepTimer("Initialize Llama(..., verbose=True)"):
            self.llama = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,  # -1 tries to offload all
                verbose=True,
            )

        info("GGUF model loaded successfully.")

    def generate(self, prompt: str, print_prompt: bool = True, print_output: bool = True) -> Dict[str, Any]:
        if self.llama is None:
            raise RuntimeError("LlamaCppRunner not loaded. Call load() first.")

        section("llama.cpp inference (ultra-verbose)")
        kv("prompt_chars", len(prompt))
        if print_prompt:
            print("\n--- Prompt (verbatim) ---")
            print(prompt)
            print("--- End prompt ---\n")

        gen_kwargs = dict(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repeat_penalty=self.repetition_penalty,
        )

        section("Generation parameters")
        for k, v in gen_kwargs.items():
            kv(k, v)

        with StepTimer("llama(prompt, ...)"):
            out = self.llama(prompt, **gen_kwargs)

        text = out["choices"][0]["text"].strip()

        # Best-effort token accounting
        usage = out.get("usage", {})
        kv("usage", usage if usage else "(not provided)")

        if print_output:
            print("\n--- Model output ---")
            print(text)
            print("--- End output ---\n")

        return {"text": text}


# =============================================================================
# Runner
# =============================================================================

def load_text_file(path: str) -> str:
    with StepTimer(f"Read file: {path}"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

def run_once(runner: Any, prompt: str) -> str:
    result = runner.generate(prompt, print_prompt=True, print_output=True)
    return result["text"]

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_opensource_LLM.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Run Tencent HY-MT1.5 translation model (1.8B) with VERY verbose debug output."
    )

    parser.add_argument("--backend", choices=["transformers", "llama_cpp"], default="transformers",
                        help="Inference backend:\n"
                             "  transformers = PyTorch/Transformers (uses MPS on Mac if available)\n"
                             "  llama_cpp   = llama.cpp via llama-cpp-python (GGUF; great on Mac)")
    parser.add_argument(
        "--model_id",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Hugging Face chat/instruction model id for conversational Q&A"
    ) # changed from 'tencent/HY-MT1.5-1.8B' to 'Qwen/Qwen2.5-1.5B-Instruct' as HY-MT1.5-1.8B is not a Q&A type LLM

    parser.add_argument("--device", default="auto", help="Device preference for transformers: auto|mps|cpu|cuda")

    # Keep CLI name --torch_dtype for backwards compatibility, but internally we treat it as "dtype choice".
    parser.add_argument("--torch_dtype", default="auto",
                        help="Dtype choice: auto|fp16|bf16|fp32\n"
                             "Note: internally this maps to either `dtype=` or `torch_dtype=` depending on your transformers version.")
    parser.add_argument("--warmup", action="store_true",
                        help="Run a tiny warmup generation after loading to validate end-to-end early.")

    # llama.cpp / GGUF
    parser.add_argument("--gguf_repo", default="tencent/HY-MT1.5-1.8B-GGUF",
                        help="Hugging Face GGUF repo id for llama_cpp backend.")
    parser.add_argument("--gguf_file", default="HY-MT1.5-1.8B-Q4_K_M.gguf",
                        help="GGUF filename in the repo (Q4_K_M is smallest and usually best on laptops).")
    parser.add_argument("--n_ctx", type=int, default=4096, help="Context window for llama.cpp backend.")
    parser.add_argument("--n_threads", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                        help="Threads for llama.cpp.")
    parser.add_argument("--n_gpu_layers", type=int, default=-1,
                        help="llama.cpp: number of layers to offload to GPU (Metal). -1 tries to offload all.")

    # generation params (recommended defaults from model card)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.6)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    # prompting
    parser.add_argument("--prompt_style", choices=["auto", "en", "zh"], default="auto",
                        help="Prompt language style for basic translation:\n"
                             "  auto = zh style if target looks like Chinese, else English\n"
                             "  en   = always English template\n"
                             "  zh   = always Chinese template")

    # input modes
    parser.add_argument("--target_language", default="Spanish", help="Target language name (e.g., Spanish, 中文).")
    parser.add_argument("--text", default=None, help="Text to translate (if omitted, use --run_tests or --interactive).")
    parser.add_argument("--text_file", default=None, help="Path to a text file to translate.")
    parser.add_argument("--run_tests", action="store_true", help="Run the built-in biomedical/mental-health test suite.")
    parser.add_argument("--interactive", action="store_true", help="Interactive loop: enter text, get translations.")
    parser.add_argument("--kind", choices=["basic", "terminology", "context", "formatted"], default="basic",
                        help="Which prompt kind to use for --text/--text_file mode.")

    # optional for terminology/context
    parser.add_argument("--source_term", default=None, help="Terminology intervention: source term.")
    parser.add_argument("--target_term", default=None, help="Terminology intervention: target term.")
    parser.add_argument("--context", default=None, help="Contextual translation: context text (not translated).")
    parser.add_argument("--context_file", default=None, help="Contextual translation: path to context file.")

    args = parser.parse_args()

    print_system_diagnostics()

    section("License reminder (read before using)")
    print("Tencent's HY-MT1.5 license text states it does NOT apply in the EU/UK/South Korea.")
    print("See: https://huggingface.co/tencent/HY-MT1.5-1.8B/blob/main/License.txt")
    print("You are responsible for ensuring you are allowed to use this model where you are.\n")

    # Load runner
    if args.backend == "transformers":
        runner = TransformersRunner(
            model_id=args.model_id,
            device_preference=args.device,
            dtype_choice=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            warmup=args.warmup,
        )
    else:
        runner = LlamaCppRunner(
            repo_id=args.gguf_repo,
            gguf_filename=args.gguf_file,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            n_gpu_layers=args.n_gpu_layers,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

    with StepTimer("runner.load()"):
        runner.load()

    # Resolve input text/context from other if needed
    if args.text_file and not args.text:
        args.text = load_text_file(args.text_file)

    if args.context_file and not args.context:
        args.context = load_text_file(args.context_file)

    # Run tests
    if args.run_tests:
        section("Running built-in test suite")
        tests = build_default_test_suite()
        for i, tc in enumerate(tests, start=1):
            section(f"Test {i}/{len(tests)} — {tc.name}")
            prompt = build_prompt(
                kind=tc.kind,
                prompt_style=args.prompt_style,
                target_language=tc.target_language,
                source_text=tc.source_text,
                source_term=tc.source_term,
                target_term=tc.target_term,
                context=tc.context,
            )
            _ = run_once(runner, prompt)

    # Single translation
    if args.text:
        section("Single translation mode (--text/--text_file)")
        prompt = build_prompt(
            kind=args.kind,
            prompt_style=args.prompt_style,
            target_language=args.target_language,
            source_text=args.text,
            source_term=args.source_term,
            target_term=args.target_term,
            context=args.context,
        )
        _ = run_once(runner, prompt)

    # Interactive loop
    if args.interactive:
        section("Interactive mode")
        print("Type/paste text to translate. End with an empty line to run. Ctrl+C to exit.\n")
        buff: List[str] = []
        try:
            while True:
                line = input()
                if line.strip() == "":
                    if not buff:
                        continue
                    source_text = "\n".join(buff).strip()
                    buff = []
                    prompt = build_prompt(
                        kind=args.kind,
                        prompt_style=args.prompt_style,
                        target_language=args.target_language,
                        source_text=source_text,
                        source_term=args.source_term,
                        target_term=args.target_term,
                        context=args.context,
                    )
                    _ = run_once(runner, prompt)
                    print("Enter next text (empty line to run):")
                else:
                    buff.append(line)
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")

    if (not args.run_tests) and (not args.text) and (not args.interactive):
        section("Nothing to do")
        print("You didn't provide --text/--text_file and didn't enable --run_tests or --interactive.")
        print("Examples:")
        print("  python run_opensource_LLM.py --run_tests --warmup")
        print("  python run_opensource_LLM.py --text \"It’s on the house.\" --target_language Spanish")
        print("  python run_opensource_LLM.py --backend llama_cpp --run_tests")
        print("  python run_opensource_LLM.py --backend llama_cpp --gguf_file HY-MT1.5-1.8B-Q8_0.gguf --run_tests")

    section("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# RUN THIS: 'python /Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/other/tempo/run_opensource_LLM.py --interactive --model_id Qwen/Qwen2.5-1.5B-Instruct'

# What the LLM currently does (i.e., task type) is translation to the target language specified by the user.
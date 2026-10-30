# piembed

Vector embedding and semantic cue matching microservice for the Raspberry Pi production ecosystem at Riihimäen Keskuskirkko.

> **Ecosystem Status Requirement**  
> All updates to this repository's status, service contracts, port allocations, or operational health MUST be documented in the centralized [piecosystem](https://github.com/keskuskirkko/piecosystem) registry.

---

## Overview

`piembed` provides fast, dimension-agnostic text embeddings and cosine similarity comparisons for real-time semantic cue detection in [`pirundown`](file:///D:/pirundown).

By default, `piembed` queries an adjacent **Ollama** server (`POST /api/embed`) to leverage dedicated GPU/NPU hardware. If the remote Ollama server is unreachable or times out, `piembed` automatically trips its circuit breaker and falls back to an **in-process Hugging Face ONNX runtime** (`@huggingface/transformers`) with zero dropped broadcast cues.

## Features

- **Remote Ollama Default + Automatic In-Process Fallback**: Zero-downtime failover between remote Ollama (`bge-m3`, `nomic-embed-text`) and local quantized ONNX models (`Xenova/paraphrase-multilingual-MiniLM-L12-v2`).
- **Dynamic Hugging Face Model Management**: Load and pre-download Hugging Face models into the local disk cache at runtime via REST API.
- **Dimension-Agnostic Cosine Engine**: Supports 384, 768, 1024, 1536, or arbitrary vector dimensions with 4x-unrolled `Float32Array` similarity math and automatic re-indexing if provider/model dimensions change mid-rundown.
- **Two-Stage Semantic Cue Protocol**:
  1. **Preflight**: Batch-embeds and caches rundown cue target phrases when a rundown is loaded (`POST /api/v1/cues/preflight`).
  2. **Live Match**: Evaluates rolling ASR caption windows against active cue IDs in a single forward pass (`POST /api/v1/cues/match`).

## API Endpoints (Port `3125`)

### Semantic Cue Workflow
- `POST /api/v1/cues/preflight` — Register and pre-compute vectors for rundown semantic cues (`{ rundownId, cues: [{ id, phrase }] }`).
- `POST /api/v1/cues/match` — Match live caption window against candidate cue IDs (`{ text, cueIds, threshold }`).
- `GET /api/v1/cues` — List registered cues and active rundowns.
- `DELETE /api/v1/cues/preflight/:rundownId` — Unload preflight vectors for a rundown.

### Model Management
- `GET /api/v1/models` — Inspect active provider, model dimensions, Ollama health, and cached Hugging Face models.
- `POST /api/v1/models/load` — Load a model from Hugging Face (`source: "huggingface"`) or Ollama (`source: "ollama"`).
- `POST /api/v1/models/download` — Pre-download a Hugging Face model into the local ONNX disk cache.

### General Utilities & Health
- `POST /api/v1/embed` — Batch-compute embeddings (`{ texts: string[] }`).
- `POST /api/v1/compare` — Direct cosine similarity comparison between `query` and `candidates`.
- `GET /health` & `GET /api/v1/health` — Service health and provider status.

## License

ISC

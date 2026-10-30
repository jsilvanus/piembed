import { l2Normalize } from "../vector/math.js";
import type { EmbeddingProvider, ProviderEmbedOutput } from "./types.js";

export interface OllamaProviderOptions {
  baseUrl: string;
  defaultModel: string;
  timeoutMs?: number;
  fetchFn?: typeof fetch;
}

export class OllamaEmbeddingProvider implements EmbeddingProvider {
  public readonly kind = "ollama" as const;
  private baseUrl: string;
  private currentModel: string;
  private timeoutMs: number;
  private lastDimensions: number | null = null;
  private fetchFn: typeof fetch;

  constructor(options: OllamaProviderOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.currentModel = options.defaultModel;
    this.timeoutMs = options.timeoutMs ?? 2500;
    this.fetchFn = options.fetchFn ?? globalThis.fetch;
  }

  getModel(): string {
    return this.currentModel;
  }

  setModel(model: string): void {
    if (model !== this.currentModel) {
      this.currentModel = model;
      this.lastDimensions = null;
    }
  }

  getDimensions(): number | null {
    return this.lastDimensions;
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  async checkHealth(): Promise<{ ok: boolean; models: string[]; error?: string }> {
    try {
      const res = await this.fetchFn(`${this.baseUrl}/api/tags`, {
        method: "GET",
        signal: AbortSignal.timeout(this.timeoutMs),
      });
      if (!res.ok) {
        return { ok: false, models: [], error: `HTTP ${res.status}` };
      }
      const data = (await res.json()) as { models?: Array<{ name?: string; model?: string }> };
      const models = (data.models ?? [])
        .map((m) => m.name ?? m.model ?? "")
        .filter(Boolean);
      return { ok: true, models };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return { ok: false, models: [], error: message };
    }
  }

  async loadModel(model: string): Promise<{ model: string; dimensions: number }> {
    this.setModel(model);
    const probe = await this.embed(["test"], model);
    return { model: probe.model, dimensions: probe.dimensions };
  }

  async embed(texts: string[], requestedModel?: string): Promise<ProviderEmbedOutput> {
    const model = requestedModel || this.currentModel;
    if (texts.length === 0) {
      return { vectors: [], model, dimensions: this.lastDimensions ?? 0 };
    }

    const res = await this.fetchFn(`${this.baseUrl}/api/embed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, input: texts }),
      signal: AbortSignal.timeout(this.timeoutMs),
    });

    if (res.status === 404) {
      // Fallback to legacy /api/embeddings (one by one)
      return this.embedLegacy(texts, model);
    }

    if (!res.ok) {
      const bodyText = await res.text().catch(() => "");
      throw new Error(`Ollama /api/embed failed with HTTP ${res.status}: ${bodyText}`);
    }

    const payload = (await res.json()) as { embeddings?: number[][] };
    if (!Array.isArray(payload.embeddings) || payload.embeddings.length !== texts.length) {
      throw new Error("Ollama /api/embed returned invalid embeddings array");
    }

    const vectors = payload.embeddings.map((raw) => l2Normalize(raw));
    const dimensions = vectors[0]?.length ?? 0;
    if (model === this.currentModel && dimensions > 0) {
      this.lastDimensions = dimensions;
    }

    return { vectors, model, dimensions };
  }

  private async embedLegacy(texts: string[], model: string): Promise<ProviderEmbedOutput> {
    const vectors: Float32Array[] = [];
    for (const prompt of texts) {
      const res = await this.fetchFn(`${this.baseUrl}/api/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, prompt }),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
      if (!res.ok) {
        throw new Error(`Ollama /api/embeddings failed with HTTP ${res.status}`);
      }
      const data = (await res.json()) as { embedding?: number[] };
      if (!Array.isArray(data.embedding)) {
        throw new Error("Ollama /api/embeddings returned invalid embedding");
      }
      vectors.push(l2Normalize(data.embedding));
    }
    const dimensions = vectors[0]?.length ?? 0;
    if (model === this.currentModel && dimensions > 0) {
      this.lastDimensions = dimensions;
    }
    return { vectors, model, dimensions };
  }
}

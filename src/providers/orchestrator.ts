import { logger } from "../system/logger.js";
import { VectorLruCache } from "../vector/lru-cache.js";
import type { OllamaEmbeddingProvider } from "./ollama-provider.js";
import type { InProcessTransformersProvider } from "./transformers-provider.js";
import type { EmbeddingResult, ProviderKind } from "./types.js";

export interface OrchestratorOptions {
  primary: OllamaEmbeddingProvider;
  fallback: InProcessTransformersProvider;
  cache?: VectorLruCache;
  healthCheckIntervalMs?: number;
  preferProvider?: ProviderKind;
}

export class EmbeddingOrchestrator {
  private readonly primary: OllamaEmbeddingProvider;
  private readonly fallback: InProcessTransformersProvider;
  private readonly cache: VectorLruCache;
  private readonly healthCheckIntervalMs: number;
  private preferredProvider: ProviderKind;
  private ollamaHealthy = true;
  private ollamaModels: string[] = [];
  private lastHealthCheckAt = 0;
  private healthTimer: NodeJS.Timeout | null = null;

  constructor(options: OrchestratorOptions) {
    this.primary = options.primary;
    this.fallback = options.fallback;
    this.cache = options.cache ?? new VectorLruCache(2000);
    this.healthCheckIntervalMs = options.healthCheckIntervalMs ?? 15000;
    this.preferredProvider = options.preferProvider ?? "ollama";
  }

  startHealthLoop(): void {
    if (this.healthTimer) return;
    void this.probeOllamaHealth();
    if (this.healthCheckIntervalMs > 0) {
      this.healthTimer = setInterval(() => {
        void this.probeOllamaHealth();
      }, this.healthCheckIntervalMs);
      this.healthTimer.unref?.();
    }
  }

  stopHealthLoop(): void {
    if (this.healthTimer) {
      clearInterval(this.healthTimer);
      this.healthTimer = null;
    }
  }

  async probeOllamaHealth(): Promise<boolean> {
    const res = await this.primary.checkHealth();
    this.lastHealthCheckAt = Date.now();
    const wasHealthy = this.ollamaHealthy;
    this.ollamaHealthy = res.ok;
    if (res.ok) {
      this.ollamaModels = res.models;
      if (!wasHealthy) {
        logger.info(`Remote Ollama at ${this.primary.getBaseUrl()} recovered and is online.`);
      }
    } else if (wasHealthy) {
      logger.warn(`Remote Ollama health check failed (${res.error ?? "unreachable"}). Will use in-process fallback.`);
    }
    return this.ollamaHealthy;
  }

  getActiveProviderKind(): ProviderKind {
    if (this.preferredProvider === "inprocess") {
      return "inprocess";
    }
    return this.ollamaHealthy ? "ollama" : "inprocess";
  }

  setPreferredProvider(provider: ProviderKind): void {
    this.preferredProvider = provider;
  }

  getActiveModel(): string {
    return this.getActiveProviderKind() === "ollama"
      ? this.primary.getModel()
      : this.fallback.getModel();
  }

  getActiveDimensions(): number | null {
    return this.getActiveProviderKind() === "ollama"
      ? this.primary.getDimensions()
      : this.fallback.getDimensions();
  }

  getPrimary(): OllamaEmbeddingProvider {
    return this.primary;
  }

  getFallback(): InProcessTransformersProvider {
    return this.fallback;
  }

  getCache(): VectorLruCache {
    return this.cache;
  }

  getStatus() {
    const activeProvider = this.getActiveProviderKind();
    return {
      activeProvider,
      preferredProvider: this.preferredProvider,
      activeModel: this.getActiveModel(),
      activeDimensions: this.getActiveDimensions(),
      ollama: {
        url: this.primary.getBaseUrl(),
        healthy: this.ollamaHealthy,
        model: this.primary.getModel(),
        dimensions: this.primary.getDimensions(),
        availableModels: this.ollamaModels,
        lastHealthCheckAt: this.lastHealthCheckAt,
      },
      inprocess: {
        loaded: this.fallback.isLoaded(),
        model: this.fallback.getModel(),
        quantized: this.fallback.getQuantized(),
        dimensions: this.fallback.getDimensions(),
        cacheDir: this.fallback.getCacheDir(),
        cachedModels: this.fallback.listCachedModels(),
      },
      cacheSize: this.cache.size,
    };
  }

  async embed(texts: string[], requestedModel?: string): Promise<EmbeddingResult> {
    const start = Date.now();
    if (texts.length === 0) {
      const provider = this.getActiveProviderKind();
      return {
        vectors: [],
        provider,
        model: requestedModel || this.getActiveModel(),
        dimensions: this.getActiveDimensions() ?? 0,
        durationMs: 0,
        cachedHits: 0,
      };
    }

    if (this.preferredProvider === "ollama" && this.ollamaHealthy) {
      try {
        const model = requestedModel || this.primary.getModel();
        const res = await this.embedWithCache("ollama", model, texts, (missing) =>
          this.primary.embed(missing, model)
        );
        return {
          ...res,
          provider: "ollama",
          durationMs: Date.now() - start,
        };
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.warn(`Remote Ollama call failed (${msg}). Tripping circuit breaker to in-process fallback.`);
        this.ollamaHealthy = false;
        this.lastHealthCheckAt = Date.now();
      }
    }

    // Fallback to in-process Hugging Face Transformers
    const fallbackModel =
      requestedModel && this.preferredProvider === "inprocess"
        ? requestedModel
        : this.fallback.getModel();
    const res = await this.embedWithCache("inprocess", fallbackModel, texts, (missing) =>
      this.fallback.embed(missing, fallbackModel)
    );
    return {
      ...res,
      provider: "inprocess",
      durationMs: Date.now() - start,
    };
  }

  private async embedWithCache(
    provider: ProviderKind,
    model: string,
    texts: string[],
    computeFn: (missingTexts: string[]) => Promise<{ vectors: Float32Array[]; model: string; dimensions: number }>
  ): Promise<{ vectors: Float32Array[]; model: string; dimensions: number; cachedHits: number }> {
    const results: Array<Float32Array | null> = new Array(texts.length).fill(null);
    const missingIndices: number[] = [];
    const missingTexts: string[] = [];
    let cachedHits = 0;

    for (let i = 0; i < texts.length; i++) {
      const hit = this.cache.get(provider, model, texts[i]);
      if (hit) {
        results[i] = hit.vector;
        cachedHits++;
      } else {
        missingIndices.push(i);
        missingTexts.push(texts[i]);
      }
    }

    let resolvedModel = model;
    let dimensions = results.find((v) => v !== null)?.length ?? 0;

    if (missingTexts.length > 0) {
      const computed = await computeFn(missingTexts);
      resolvedModel = computed.model;
      dimensions = computed.dimensions;
      for (let k = 0; k < missingIndices.length; k++) {
        const idx = missingIndices[k];
        const vec = computed.vectors[k];
        results[idx] = vec;
        if (vec) {
          this.cache.set(provider, resolvedModel, missingTexts[k], vec);
        }
      }
    }

    return {
      vectors: results.map((v) => v ?? new Float32Array(dimensions)),
      model: resolvedModel,
      dimensions,
      cachedHits,
    };
  }
}

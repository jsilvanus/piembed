import fs from "node:fs";
import path from "node:path";
import { l2Normalize } from "../vector/math.js";
import type { EmbeddingProvider, ProviderEmbedOutput } from "./types.js";

export type FeatureExtractionPipeline = (
  texts: string | string[],
  options?: { pooling?: "mean" | "cls" | "none"; normalize?: boolean }
) => Promise<{ data: Float32Array | number[]; dims?: number[]; tolist?: () => unknown }>;

export type PipelineLoaderFn = (
  modelId: string,
  options: { quantized: boolean; cacheDir: string }
) => Promise<FeatureExtractionPipeline>;

export interface TransformersProviderOptions {
  defaultModel: string;
  quantized?: boolean;
  cacheDir: string;
  pipelineLoader?: PipelineLoaderFn;
}

const defaultPipelineLoader: PipelineLoaderFn = async (modelId, options) => {
  const { pipeline } = await import("@huggingface/transformers");
  const pipe = await pipeline("feature-extraction", modelId, {
    dtype: options.quantized ? "q8" : "fp32",
    cache_dir: options.cacheDir,
  });
  return pipe as unknown as FeatureExtractionPipeline;
};

export class InProcessTransformersProvider implements EmbeddingProvider {
  public readonly kind = "inprocess" as const;
  private currentModel: string;
  private quantized: boolean;
  private cacheDir: string;
  private pipeline: FeatureExtractionPipeline | null = null;
  private loadingPromise: Promise<FeatureExtractionPipeline> | null = null;
  private lastDimensions: number | null = null;
  private pipelineLoader: PipelineLoaderFn;
  private downloadedModels = new Set<string>();

  constructor(options: TransformersProviderOptions) {
    this.currentModel = options.defaultModel;
    this.quantized = options.quantized ?? true;
    this.cacheDir = options.cacheDir;
    this.pipelineLoader = options.pipelineLoader ?? defaultPipelineLoader;
  }

  getModel(): string {
    return this.currentModel;
  }

  getQuantized(): boolean {
    return this.quantized;
  }

  getDimensions(): number | null {
    return this.lastDimensions;
  }

  isLoaded(): boolean {
    return this.pipeline !== null;
  }

  getCacheDir(): string {
    return this.cacheDir;
  }

  listCachedModels(): string[] {
    const models = new Set<string>(this.downloadedModels);
    try {
      if (fs.existsSync(this.cacheDir)) {
        const orgs = fs.readdirSync(this.cacheDir, { withFileTypes: true });
        for (const org of orgs) {
          if (org.isDirectory()) {
            const subPath = path.join(this.cacheDir, org.name);
            const repos = fs.readdirSync(subPath, { withFileTypes: true });
            for (const repo of repos) {
              if (repo.isDirectory()) {
                models.add(`${org.name}/${repo.name}`);
              }
            }
          }
        }
      }
    } catch {
      // Ignore fs errors on non-existent cache directory
    }
    return Array.from(models);
  }

  async loadModel(
    modelId: string,
    options?: { quantized?: boolean }
  ): Promise<{ model: string; dimensions: number; quantized: boolean }> {
    const quantized = options?.quantized ?? this.quantized;
    fs.mkdirSync(this.cacheDir, { recursive: true });

    const pipe = await this.pipelineLoader(modelId, {
      quantized,
      cacheDir: this.cacheDir,
    });

    this.pipeline = pipe;
    this.currentModel = modelId;
    this.quantized = quantized;
    this.downloadedModels.add(modelId);

    const probe = await this.extractVectors(pipe, ["test"]);
    const dimensions = probe[0]?.length ?? 0;
    this.lastDimensions = dimensions;

    return { model: modelId, dimensions, quantized };
  }

  async downloadModel(
    modelId: string,
    options?: { quantized?: boolean }
  ): Promise<{ model: string; dimensions: number; quantized: boolean; cached: boolean }> {
    const quantized = options?.quantized ?? this.quantized;
    fs.mkdirSync(this.cacheDir, { recursive: true });

    const pipe = await this.pipelineLoader(modelId, {
      quantized,
      cacheDir: this.cacheDir,
    });

    this.downloadedModels.add(modelId);
    const probe = await this.extractVectors(pipe, ["test"]);
    const dimensions = probe[0]?.length ?? 0;

    // If no pipeline is currently loaded, adopt this one
    if (!this.pipeline) {
      this.pipeline = pipe;
      this.currentModel = modelId;
      this.quantized = quantized;
      this.lastDimensions = dimensions;
    }

    return { model: modelId, dimensions, quantized, cached: true };
  }

  private async ensurePipeline(requestedModel?: string): Promise<FeatureExtractionPipeline> {
    const targetModel = requestedModel || this.currentModel;
    if (this.pipeline && targetModel === this.currentModel) {
      return this.pipeline;
    }
    if (this.loadingPromise && targetModel === this.currentModel) {
      return this.loadingPromise;
    }

    this.loadingPromise = (async () => {
      try {
        await this.loadModel(targetModel, { quantized: this.quantized });
        return this.pipeline!;
      } finally {
        this.loadingPromise = null;
      }
    })();

    return this.loadingPromise;
  }

  async embed(texts: string[], requestedModel?: string): Promise<ProviderEmbedOutput> {
    const model = requestedModel || this.currentModel;
    if (texts.length === 0) {
      return { vectors: [], model, dimensions: this.lastDimensions ?? 0 };
    }

    const pipe = await this.ensurePipeline(requestedModel);
    const vectors = await this.extractVectors(pipe, texts);
    const dimensions = vectors[0]?.length ?? 0;
    if (dimensions > 0) {
      this.lastDimensions = dimensions;
    }

    return { vectors, model: this.currentModel, dimensions };
  }

  private async extractVectors(
    pipe: FeatureExtractionPipeline,
    texts: string[]
  ): Promise<Float32Array[]> {
    const output = await pipe(texts, { pooling: "mean", normalize: true });
    const rawData = output.data;
    const totalLen = rawData.length;
    const batchSize = texts.length;

    if (batchSize === 0 || totalLen === 0) {
      return [];
    }

    const dim = Math.floor(totalLen / batchSize);
    const results: Float32Array[] = [];

    for (let i = 0; i < batchSize; i++) {
      const slice = new Float32Array(dim);
      const offset = i * dim;
      for (let j = 0; j < dim; j++) {
        slice[j] = rawData[offset + j];
      }
      results.push(l2Normalize(slice));
    }

    return results;
  }
}

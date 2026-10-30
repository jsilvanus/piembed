export type ProviderKind = "ollama" | "inprocess";

export interface ProviderEmbedOutput {
  vectors: Float32Array[];
  model: string;
  dimensions: number;
}

export interface EmbeddingResult extends ProviderEmbedOutput {
  provider: ProviderKind;
  durationMs: number;
  cachedHits?: number;
}

export interface EmbeddingProvider {
  readonly kind: ProviderKind;
  getModel(): string;
  getDimensions(): number | null;
  embed(texts: string[], requestedModel?: string): Promise<ProviderEmbedOutput>;
}

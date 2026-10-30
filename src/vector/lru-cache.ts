export interface CachedVectorEntry {
  vector: Float32Array;
  model: string;
  provider: "ollama" | "inprocess";
  dimensions: number;
  createdAt: number;
}

/**
 * Model-tagged LRU cache for phrase embeddings.
 * Keys are composite `${provider}:${model}:${text}` so switching models or providers
 * never collides with stale dimensions.
 */
export class VectorLruCache {
  private readonly map = new Map<string, CachedVectorEntry>();

  constructor(private readonly maxEntries: number = 2000) {}

  private makeKey(provider: string, model: string, text: string): string {
    return `${provider}::${model}::${text.trim()}`;
  }

  get(provider: string, model: string, text: string): CachedVectorEntry | undefined {
    const key = this.makeKey(provider, model, text);
    const entry = this.map.get(key);
    if (!entry) {
      return undefined;
    }
    // Refresh LRU order
    this.map.delete(key);
    this.map.set(key, entry);
    return entry;
  }

  set(provider: "ollama" | "inprocess", model: string, text: string, vector: Float32Array): CachedVectorEntry {
    const key = this.makeKey(provider, model, text);
    if (this.map.has(key)) {
      this.map.delete(key);
    } else if (this.map.size >= this.maxEntries) {
      const oldestKey = this.map.keys().next().value;
      if (oldestKey !== undefined) {
        this.map.delete(oldestKey);
      }
    }
    const entry: CachedVectorEntry = {
      vector,
      model,
      provider,
      dimensions: vector.length,
      createdAt: Date.now(),
    };
    this.map.set(key, entry);
    return entry;
  }

  clear(): void {
    this.map.clear();
  }

  get size(): number {
    return this.map.size;
  }
}

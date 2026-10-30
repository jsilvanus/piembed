export interface CueInput {
  id: string;
  phrase: string;
}

export interface StoredCue {
  id: string;
  rundownId: string;
  phrase: string;
  vector: Float32Array;
  model: string;
  provider: "ollama" | "inprocess";
  dimensions: number;
  updatedAt: number;
}

/**
 * In-memory registry of preflighted semantic cues.
 * Stores cue metadata alongside its pre-computed vector, and detects if the active
 * embedding provider/model changes so cues can be lazily re-embedded on the fly.
 */
export class CueStore {
  private readonly cuesById = new Map<string, StoredCue>();
  private readonly rundownCueIds = new Map<string, Set<string>>();

  registerBatch(
    rundownId: string,
    cues: CueInput[],
    vectors: Float32Array[],
    model: string,
    provider: "ollama" | "inprocess"
  ): StoredCue[] {
    const now = Date.now();
    let set = this.rundownCueIds.get(rundownId);
    if (!set) {
      set = new Set<string>();
      this.rundownCueIds.set(rundownId, set);
    }

    const stored: StoredCue[] = [];
    for (let i = 0; i < cues.length; i++) {
      const cue = cues[i];
      const vector = vectors[i];
      if (!cue || !vector) continue;

      const entry: StoredCue = {
        id: cue.id,
        rundownId,
        phrase: cue.phrase,
        vector,
        model,
        provider,
        dimensions: vector.length,
        updatedAt: now,
      };
      this.cuesById.set(cue.id, entry);
      set.add(cue.id);
      stored.push(entry);
    }
    return stored;
  }

  getCue(id: string): StoredCue | undefined {
    return this.cuesById.get(id);
  }

  getCues(cueIds?: string[]): StoredCue[] {
    if (!cueIds || cueIds.length === 0) {
      return Array.from(this.cuesById.values());
    }
    const list: StoredCue[] = [];
    for (const id of cueIds) {
      const entry = this.cuesById.get(id);
      if (entry) {
        list.push(entry);
      }
    }
    return list;
  }

  deleteRundown(rundownId: string): number {
    const set = this.rundownCueIds.get(rundownId);
    if (!set) {
      return 0;
    }
    let removed = 0;
    for (const id of set) {
      const entry = this.cuesById.get(id);
      if (entry && entry.rundownId === rundownId) {
        this.cuesById.delete(id);
        removed++;
      }
    }
    this.rundownCueIds.delete(rundownId);
    return removed;
  }

  clear(): void {
    this.cuesById.clear();
    this.rundownCueIds.clear();
  }

  get size(): number {
    return this.cuesById.size;
  }

  listRundowns(): Array<{ rundownId: string; cueCount: number }> {
    const out: Array<{ rundownId: string; cueCount: number }> = [];
    for (const [rundownId, ids] of this.rundownCueIds.entries()) {
      out.push({ rundownId, cueCount: ids.size });
    }
    return out;
  }
}

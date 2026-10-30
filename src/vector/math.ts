export class DimensionMismatchError extends Error {
  constructor(public readonly dimA: number, public readonly dimB: number) {
    super(`Vector dimension mismatch: cannot compare ${dimA}-d vector with ${dimB}-d vector`);
    this.name = "DimensionMismatchError";
  }
}

export function toFloat32Array(vec: Float32Array | number[]): Float32Array {
  if (vec instanceof Float32Array) {
    return vec;
  }
  return new Float32Array(vec);
}

export function l2Norm(vec: Float32Array | number[]): number {
  let sum = 0;
  const len = vec.length;
  for (let i = 0; i < len; i++) {
    const v = vec[i];
    sum += v * v;
  }
  return Math.sqrt(sum);
}

export function l2Normalize(vec: Float32Array | number[]): Float32Array {
  const out = new Float32Array(vec.length);
  const norm = l2Norm(vec);
  if (norm === 0) {
    return out;
  }
  const inv = 1 / norm;
  for (let i = 0; i < vec.length; i++) {
    out[i] = vec[i] * inv;
  }
  return out;
}

/**
 * Dimension-agnostic cosine similarity between two vectors.
 * Unrolled 4x for V8 optimization.
 */
export function cosineSimilarity(a: Float32Array | number[], b: Float32Array | number[]): number {
  const len = a.length;
  if (len !== b.length) {
    throw new DimensionMismatchError(len, b.length);
  }
  if (len === 0) {
    return 0;
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  let i = 0;
  const limit = len - (len % 4);
  for (; i < limit; i += 4) {
    const a0 = a[i], b0 = b[i];
    const a1 = a[i + 1], b1 = b[i + 1];
    const a2 = a[i + 2], b2 = b[i + 2];
    const a3 = a[i + 3], b3 = b[i + 3];

    dot += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    normA += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
    normB += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
  }

  for (; i < len; i++) {
    const ai = a[i];
    const bi = b[i];
    dot += ai * bi;
    normA += ai * ai;
    normB += bi * bi;
  }

  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom === 0) {
    return 0;
  }
  return dot / denom;
}

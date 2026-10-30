import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  cosineSimilarity,
  DimensionMismatchError,
  l2Norm,
  l2Normalize,
} from "../src/vector/math.js";

describe("Vector Math & Dimension Agnosticism", () => {
  it("computes cosine similarity accurately across 384, 768, and 1024 dimensions", () => {
    for (const dim of [384, 768, 1024, 1536]) {
      const a = new Float32Array(dim);
      const b = new Float32Array(dim);
      for (let i = 0; i < dim; i++) {
        a[i] = Math.sin(i + 1);
        b[i] = Math.sin(i + 1);
      }

      const simIdentical = cosineSimilarity(a, b);
      assert.ok(Math.abs(simIdentical - 1.0) < 1e-5, `Expected ~1.0 for dim ${dim}, got ${simIdentical}`);

      const c = new Float32Array(dim);
      for (let i = 0; i < dim; i++) {
        c[i] = -a[i];
      }
      const simOpposite = cosineSimilarity(a, c);
      assert.ok(Math.abs(simOpposite + 1.0) < 1e-5, `Expected ~-1.0 for dim ${dim}, got ${simOpposite}`);
    }
  });

  it("normalizes vectors to unit L2 length", () => {
    const raw = [3, 4, 0, 0, 0];
    const norm = l2Normalize(raw);
    assert.ok(Math.abs(l2Norm(norm) - 1.0) < 1e-6);
    assert.ok(Math.abs(norm[0] - 0.6) < 1e-6);
    assert.ok(Math.abs(norm[1] - 0.8) < 1e-6);
  });

  it("throws DimensionMismatchError when comparing vectors of different lengths", () => {
    const v384 = new Float32Array(384);
    const v768 = new Float32Array(768);
    assert.throws(
      () => cosineSimilarity(v384, v768),
      (err: unknown) => err instanceof DimensionMismatchError && err.dimA === 384 && err.dimB === 768
    );
  });

  it("returns 0 for zero vectors without NaN", () => {
    const zero = new Float32Array(12);
    const nonZero = new Float32Array(12).fill(1);
    assert.equal(cosineSimilarity(zero, nonZero), 0);
    assert.equal(cosineSimilarity([], []), 0);
  });
});

import type { FastifyInstance } from "fastify";
import type { EmbeddingOrchestrator } from "../providers/orchestrator.js";
import { cosineSimilarity } from "../vector/math.js";

export interface EmbedBody {
  texts?: string[] | string;
  input?: string[] | string;
  model?: string;
}

export interface CompareBody {
  query?: string;
  candidates?: string[];
  model?: string;
  threshold?: number;
}

export function registerEmbedRoutes(
  app: FastifyInstance,
  orchestrator: EmbeddingOrchestrator,
  defaultThreshold: number
): void {
  // POST /api/v1/embed
  app.post<{ Body: EmbedBody }>("/api/v1/embed", async (request, reply) => {
    const body = request.body ?? {};
    const rawInput = body.texts ?? body.input;
    const texts = Array.isArray(rawInput)
      ? rawInput.map((t) => String(t))
      : typeof rawInput === "string"
        ? [rawInput]
        : [];

    if (texts.length === 0) {
      return reply.code(400).send({
        error: "Bad Request",
        message: "Field 'texts' (string or string[]) is required and must be non-empty",
      });
    }

    const res = await orchestrator.embed(texts, body.model);
    return {
      provider: res.provider,
      model: res.model,
      dimensions: res.dimensions,
      vectors: res.vectors.map((v) => Array.from(v)),
      durationMs: res.durationMs,
    };
  });

  // POST /api/v1/compare
  app.post<{ Body: CompareBody }>("/api/v1/compare", async (request, reply) => {
    const start = Date.now();
    const body = request.body ?? {};
    const query = (body.query ?? "").trim();
    const candidates = Array.isArray(body.candidates)
      ? body.candidates.map((c) => String(c).trim()).filter(Boolean)
      : [];
    const threshold =
      typeof body.threshold === "number" && Number.isFinite(body.threshold)
        ? body.threshold
        : defaultThreshold;

    if (!query || candidates.length === 0) {
      return reply.code(400).send({
        error: "Bad Request",
        message: "Fields 'query' and 'candidates' (non-empty array) are required",
      });
    }

    const res = await orchestrator.embed([query, ...candidates], body.model);
    const queryVec = res.vectors[0];
    const results = candidates.map((candidate, idx) => {
      const candidateVec = res.vectors[idx + 1];
      const rawScore = cosineSimilarity(queryVec, candidateVec);
      const score = Number(rawScore.toFixed(6));
      return {
        index: idx,
        text: candidate,
        score,
        matched: score >= threshold,
      };
    });

    results.sort((a, b) => b.score - a.score);
    const top = results[0];

    return {
      matched: Boolean(top && top.matched),
      bestMatch: top ?? null,
      results,
      provider: res.provider,
      model: res.model,
      dimensions: res.dimensions,
      latencyMs: Date.now() - start,
    };
  });
}

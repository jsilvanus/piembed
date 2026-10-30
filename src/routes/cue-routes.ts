import type { FastifyInstance } from "fastify";
import type { EmbeddingOrchestrator } from "../providers/orchestrator.js";
import type { CueInput, CueStore, StoredCue } from "../vector/cue-store.js";
import { cosineSimilarity } from "../vector/math.js";

export interface PreflightBody {
  rundownId?: string;
  cues?: CueInput[];
}

export interface MatchBody {
  text?: string;
  cueIds?: string[];
  threshold?: number;
}

export function registerCueRoutes(
  app: FastifyInstance,
  orchestrator: EmbeddingOrchestrator,
  cueStore: CueStore,
  defaultThreshold: number
): void {
  // POST /api/v1/cues/preflight
  app.post<{ Body: PreflightBody }>("/api/v1/cues/preflight", async (request, reply) => {
    const body = request.body ?? {};
    const rundownId = (body.rundownId ?? "default").trim() || "default";
    const cues = Array.isArray(body.cues)
      ? body.cues.filter((c) => c && typeof c.id === "string" && typeof c.phrase === "string" && c.phrase.trim().length > 0)
      : [];

    if (cues.length === 0) {
      return {
        status: "ok",
        rundownId,
        provider: orchestrator.getActiveProviderKind(),
        model: orchestrator.getActiveModel(),
        dimensions: orchestrator.getActiveDimensions() ?? 0,
        registered: 0,
        durationMs: 0,
      };
    }

    const phrases = cues.map((c) => c.phrase.trim());
    const embedRes = await orchestrator.embed(phrases);

    const stored = cueStore.registerBatch(
      rundownId,
      cues,
      embedRes.vectors,
      embedRes.model,
      embedRes.provider
    );

    return reply.code(200).send({
      status: "ok",
      rundownId,
      provider: embedRes.provider,
      model: embedRes.model,
      dimensions: embedRes.dimensions,
      registered: stored.length,
      durationMs: embedRes.durationMs,
    });
  });

  // POST /api/v1/cues/match
  app.post<{ Body: MatchBody }>("/api/v1/cues/match", async (request, reply) => {
    const start = Date.now();
    const body = request.body ?? {};
    const text = (body.text ?? "").trim();
    const threshold =
      typeof body.threshold === "number" && Number.isFinite(body.threshold)
        ? body.threshold
        : defaultThreshold;

    if (!text) {
      return reply.code(400).send({
        error: "Bad Request",
        message: "Field 'text' is required and must be non-empty",
      });
    }

    const candidates = cueStore.getCues(body.cueIds);
    if (candidates.length === 0) {
      return {
        matched: false,
        bestMatch: null,
        results: [],
        provider: orchestrator.getActiveProviderKind(),
        model: orchestrator.getActiveModel(),
        latencyMs: Date.now() - start,
      };
    }

    // 1. Embed incoming caption window
    const queryEmbed = await orchestrator.embed([text]);
    const queryVec = queryEmbed.vectors[0];

    // 2. Check if any candidate cues need re-embedding due to provider/model failover or model switch
    const staleCues: StoredCue[] = [];
    for (const cue of candidates) {
      if (
        cue.provider !== queryEmbed.provider ||
        cue.model !== queryEmbed.model ||
        cue.dimensions !== queryEmbed.dimensions
      ) {
        staleCues.push(cue);
      }
    }

    if (staleCues.length > 0) {
      const reEmbed = await orchestrator.embed(staleCues.map((c) => c.phrase));
      for (let i = 0; i < staleCues.length; i++) {
        const cue = staleCues[i];
        const vec = reEmbed.vectors[i];
        if (vec) {
          cueStore.registerBatch(
            cue.rundownId,
            [{ id: cue.id, phrase: cue.phrase }],
            [vec],
            reEmbed.model,
            reEmbed.provider
          );
          cue.vector = vec;
          cue.model = reEmbed.model;
          cue.provider = reEmbed.provider;
          cue.dimensions = reEmbed.dimensions;
        }
      }
    }

    // 3. Compute cosine similarity across candidates
    const results = candidates.map((cue) => {
      const rawScore = cosineSimilarity(queryVec, cue.vector);
      const score = Number(rawScore.toFixed(6));
      return {
        id: cue.id,
        phrase: cue.phrase,
        score,
        matched: score >= threshold,
      };
    });

    results.sort((a, b) => b.score - a.score);
    const top = results[0];
    const matched = Boolean(top && top.matched);

    return {
      matched,
      bestMatch: top
        ? {
            id: top.id,
            phrase: top.phrase,
            score: top.score,
            threshold,
            matched: top.matched,
          }
        : null,
      results,
      provider: queryEmbed.provider,
      model: queryEmbed.model,
      dimensions: queryEmbed.dimensions,
      latencyMs: Date.now() - start,
    };
  });

  // GET /api/v1/cues
  app.get("/api/v1/cues", async () => {
    const cues = cueStore.getCues().map((c) => ({
      id: c.id,
      rundownId: c.rundownId,
      phrase: c.phrase,
      model: c.model,
      provider: c.provider,
      dimensions: c.dimensions,
      updatedAt: c.updatedAt,
    }));
    return {
      total: cues.length,
      rundowns: cueStore.listRundowns(),
      cues,
    };
  });

  // DELETE /api/v1/cues/preflight/:rundownId
  app.delete<{ Params: { rundownId: string } }>(
    "/api/v1/cues/preflight/:rundownId",
    async (request) => {
      const { rundownId } = request.params;
      const removed = cueStore.deleteRundown(rundownId);
      return {
        status: "ok",
        rundownId,
        removed,
      };
    }
  );
}

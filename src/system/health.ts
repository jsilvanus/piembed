import type { FastifyInstance } from "fastify";
import type { EmbeddingOrchestrator } from "../providers/orchestrator.js";
import type { CueStore } from "../vector/cue-store.js";

export function registerHealthRoutes(
  app: FastifyInstance,
  orchestrator: EmbeddingOrchestrator,
  cueStore: CueStore
): void {
  const handler = async () => {
    const status = orchestrator.getStatus();
    return {
      status: "ok",
      service: "piembed",
      version: "0.1.0",
      timestamp: new Date().toISOString(),
      activeProvider: status.activeProvider,
      activeModel: status.activeModel,
      activeDimensions: status.activeDimensions,
      ollamaHealthy: status.ollama.healthy,
      inprocessLoaded: status.inprocess.loaded,
      registeredCues: cueStore.size,
      cacheSize: status.cacheSize,
    };
  };

  app.get("/health", handler);
  app.get("/api/v1/health", handler);
}

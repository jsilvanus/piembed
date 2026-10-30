import type { FastifyInstance } from "fastify";
import type { EmbeddingOrchestrator } from "../providers/orchestrator.js";

export interface LoadModelBody {
  source?: "huggingface" | "inprocess" | "ollama";
  model?: string;
  quantized?: boolean;
  activate?: boolean;
}

export interface DownloadModelBody {
  model?: string;
  quantized?: boolean;
}

export function registerModelRoutes(
  app: FastifyInstance,
  orchestrator: EmbeddingOrchestrator
): void {
  // GET /api/v1/models
  app.get("/api/v1/models", async () => {
    await orchestrator.probeOllamaHealth();
    return orchestrator.getStatus();
  });

  // POST /api/v1/models/load
  app.post<{ Body: LoadModelBody }>("/api/v1/models/load", async (request, reply) => {
    const body = request.body ?? {};
    const model = (body.model ?? "").trim();
    if (!model) {
      return reply.code(400).send({
        error: "Bad Request",
        message: "Field 'model' is required",
      });
    }

    const source =
      body.source ?? (model.includes("/") ? "huggingface" : "ollama");

    try {
      if (source === "ollama") {
        const loaded = await orchestrator.getPrimary().loadModel(model);
        if (body.activate !== false) {
          orchestrator.setPreferredProvider("ollama");
        }
        return {
          status: "ok",
          source: "ollama",
          model: loaded.model,
          dimensions: loaded.dimensions,
          activeProvider: orchestrator.getActiveProviderKind(),
        };
      }

      // Hugging Face / in-process model load
      const loaded = await orchestrator.getFallback().loadModel(model, {
        quantized: body.quantized,
      });
      if (body.activate === true) {
        orchestrator.setPreferredProvider("inprocess");
      }
      return {
        status: "ok",
        source: "huggingface",
        model: loaded.model,
        dimensions: loaded.dimensions,
        quantized: loaded.quantized,
        activeProvider: orchestrator.getActiveProviderKind(),
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return reply.code(502).send({
        error: "ModelLoadError",
        source,
        model,
        message,
      });
    }
  });

  // POST /api/v1/models/download
  app.post<{ Body: DownloadModelBody }>("/api/v1/models/download", async (request, reply) => {
    const body = request.body ?? {};
    const model = (body.model ?? "").trim();
    if (!model) {
      return reply.code(400).send({
        error: "Bad Request",
        message: "Field 'model' is required",
      });
    }

    try {
      const res = await orchestrator.getFallback().downloadModel(model, {
        quantized: body.quantized,
      });
      return {
        status: "ok",
        source: "huggingface",
        model: res.model,
        dimensions: res.dimensions,
        quantized: res.quantized,
        cached: res.cached,
        cacheDir: orchestrator.getFallback().getCacheDir(),
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return reply.code(502).send({
        error: "ModelDownloadError",
        source: "huggingface",
        model,
        message,
      });
    }
  });
}

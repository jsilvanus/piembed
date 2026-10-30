import Fastify, { type FastifyInstance } from "fastify";
import { loadConfig, type PiEmbedConfig } from "./config.js";
import { OllamaEmbeddingProvider } from "./providers/ollama-provider.js";
import { EmbeddingOrchestrator } from "./providers/orchestrator.js";
import { InProcessTransformersProvider } from "./providers/transformers-provider.js";
import { registerCueRoutes } from "./routes/cue-routes.js";
import { registerEmbedRoutes } from "./routes/embed-routes.js";
import { registerModelRoutes } from "./routes/model-routes.js";
import { registerHealthRoutes } from "./system/health.js";
import { CueStore } from "./vector/cue-store.js";
import { VectorLruCache } from "./vector/lru-cache.js";

export interface AppOptions {
  config?: Partial<PiEmbedConfig>;
  orchestrator?: EmbeddingOrchestrator;
  cueStore?: CueStore;
  startHealthLoop?: boolean;
}

export function createApp(options: AppOptions = {}): {
  app: FastifyInstance;
  config: PiEmbedConfig;
  orchestrator: EmbeddingOrchestrator;
  cueStore: CueStore;
} {
  const config: PiEmbedConfig = {
    ...loadConfig(),
    ...options.config,
  };

  const cueStore = options.cueStore ?? new CueStore();

  const orchestrator =
    options.orchestrator ??
    new EmbeddingOrchestrator({
      primary: new OllamaEmbeddingProvider({
        baseUrl: config.ollamaUrl,
        defaultModel: config.ollamaModel,
        timeoutMs: config.ollamaTimeoutMs,
      }),
      fallback: new InProcessTransformersProvider({
        defaultModel: config.inprocessModel,
        quantized: config.inprocessQuantized,
        cacheDir: config.modelCacheDir,
      }),
      cache: new VectorLruCache(config.lruMaxEntries),
      healthCheckIntervalMs: config.healthCheckIntervalMs,
    });

  if (options.startHealthLoop !== false) {
    orchestrator.startHealthLoop();
  }

  const app = Fastify({ logger: false });

  app.addHook("onClose", async () => {
    orchestrator.stopHealthLoop();
  });

  registerHealthRoutes(app, orchestrator, cueStore);
  registerCueRoutes(app, orchestrator, cueStore, config.defaultThreshold);
  registerModelRoutes(app, orchestrator);
  registerEmbedRoutes(app, orchestrator, config.defaultThreshold);

  return { app, config, orchestrator, cueStore };
}

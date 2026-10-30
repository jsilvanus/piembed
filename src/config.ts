import path from "node:path";
import os from "node:os";

export interface PiEmbedConfig {
  port: number;
  host: string;
  ollamaUrl: string;
  ollamaModel: string;
  ollamaTimeoutMs: number;
  healthCheckIntervalMs: number;
  inprocessModel: string;
  inprocessQuantized: boolean;
  modelCacheDir: string;
  defaultThreshold: number;
  lruMaxEntries: number;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): PiEmbedConfig {
  const port = Number.parseInt(env.PIEMBED_PORT ?? env.PORT ?? "3125", 10);
  const host = env.PIEMBED_HOST ?? env.HOST ?? "0.0.0.0";
  const ollamaUrl = (env.PIEMBED_OLLAMA_URL ?? env.OLLAMA_URL ?? env.OLLAMA_BASE_URL ?? "http://127.0.0.1:11434").replace(/\/+$/, "");
  const ollamaModel = env.PIEMBED_OLLAMA_MODEL ?? env.OLLAMA_EMBED_MODEL ?? "bge-m3:latest";
  const ollamaTimeoutMs = Number.parseInt(env.PIEMBED_OLLAMA_TIMEOUT_MS ?? "2500", 10);
  const healthCheckIntervalMs = Number.parseInt(env.PIEMBED_HEALTH_CHECK_INTERVAL_MS ?? "15000", 10);
  const inprocessModel = env.PIEMBED_INPROCESS_MODEL ?? "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
  const inprocessQuantized = (env.PIEMBED_INPROCESS_QUANTIZED ?? "true").toLowerCase() !== "false";
  const modelCacheDir = env.PIEMBED_MODEL_CACHE_DIR ?? path.join(os.homedir(), ".cache", "piembed", "models");
  const defaultThreshold = Number.parseFloat(env.PIEMBED_DEFAULT_THRESHOLD ?? "0.70");
  const lruMaxEntries = Number.parseInt(env.PIEMBED_LRU_MAX_ENTRIES ?? "2000", 10);

  return {
    port: Number.isFinite(port) ? port : 3125,
    host,
    ollamaUrl,
    ollamaModel,
    ollamaTimeoutMs: Number.isFinite(ollamaTimeoutMs) ? ollamaTimeoutMs : 2500,
    healthCheckIntervalMs: Number.isFinite(healthCheckIntervalMs) ? healthCheckIntervalMs : 15000,
    inprocessModel,
    inprocessQuantized,
    modelCacheDir,
    defaultThreshold: Number.isFinite(defaultThreshold) ? defaultThreshold : 0.70,
    lruMaxEntries: Number.isFinite(lruMaxEntries) ? lruMaxEntries : 2000,
  };
}

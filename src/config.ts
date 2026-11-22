import path from "node:path";
import os from "node:os";

export interface PiEmbedConfig {
  port: number;
  host: string;
  ollamaUrl: string;
  ollamaModel: string;
  ollamaTimeoutMs: number;
  /** Ollama keep_alive (PIEMBED_OLLAMA_KEEP_ALIVE, default 15m). */
  ollamaKeepAlive: string | number;
  healthCheckIntervalMs: number;
  inprocessModel: string;
  inprocessQuantized: boolean;
  modelCacheDir: string;
  defaultThreshold: number;
  lruMaxEntries: number;
}

/** Ollama `keep_alive` default: keep a model loaded for 15 minutes after its last request. */
export const DEFAULT_OLLAMA_KEEP_ALIVE = "15m";

/**
 * Turn an env/config value into what Ollama accepts: a duration string ("15m") or a number of
 * seconds (-1 = forever, 0 = unload at once). Empty/undefined gives the 15 minute default.
 */
export function normalizeKeepAlive(value: string | number | undefined | null): string | number {
  if (value === undefined || value === null) return DEFAULT_OLLAMA_KEEP_ALIVE;
  if (typeof value === "number") return value;
  const v = value.trim();
  if (!v) return DEFAULT_OLLAMA_KEEP_ALIVE;
  return /^-?\d+$/.test(v) ? Number(v) : v;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): PiEmbedConfig {
  const port = Number.parseInt(env.PIEMBED_PORT ?? env.PORT ?? "3300", 10);
  const host = env.PIEMBED_HOST ?? env.HOST ?? "0.0.0.0";
  const ollamaUrl = (env.PIEMBED_OLLAMA_URL ?? env.OLLAMA_URL ?? env.OLLAMA_BASE_URL ?? "http://127.0.0.1:11434").replace(/\/+$/, "");
  const ollamaModel = env.PIEMBED_OLLAMA_MODEL ?? env.OLLAMA_EMBED_MODEL ?? "bge-m3:latest";
  const ollamaTimeoutMs = Number.parseInt(env.PIEMBED_OLLAMA_TIMEOUT_MS ?? "2500", 10);
  const ollamaKeepAlive = normalizeKeepAlive(env.PIEMBED_OLLAMA_KEEP_ALIVE ?? env.OLLAMA_KEEP_ALIVE);
  const healthCheckIntervalMs = Number.parseInt(env.PIEMBED_HEALTH_CHECK_INTERVAL_MS ?? "15000", 10);
  const inprocessModel = env.PIEMBED_INPROCESS_MODEL ?? "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
  const inprocessQuantized = (env.PIEMBED_INPROCESS_QUANTIZED ?? "true").toLowerCase() !== "false";
  const modelCacheDir = env.PIEMBED_MODEL_CACHE_DIR ?? path.join(os.homedir(), ".cache", "piembed", "models");
  const defaultThreshold = Number.parseFloat(env.PIEMBED_DEFAULT_THRESHOLD ?? "0.70");
  const lruMaxEntries = Number.parseInt(env.PIEMBED_LRU_MAX_ENTRIES ?? "2000", 10);

  return {
    port: Number.isFinite(port) ? port : 3300,
    host,
    ollamaUrl,
    ollamaModel,
    ollamaKeepAlive,
    ollamaTimeoutMs: Number.isFinite(ollamaTimeoutMs) ? ollamaTimeoutMs : 2500,
    healthCheckIntervalMs: Number.isFinite(healthCheckIntervalMs) ? healthCheckIntervalMs : 15000,
    inprocessModel,
    inprocessQuantized,
    modelCacheDir,
    defaultThreshold: Number.isFinite(defaultThreshold) ? defaultThreshold : 0.70,
    lruMaxEntries: Number.isFinite(lruMaxEntries) ? lruMaxEntries : 2000,
  };
}

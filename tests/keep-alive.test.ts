import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { loadConfig } from "../src/config.js";
import { OllamaEmbeddingProvider } from "../src/providers/ollama-provider.js";

describe("Ollama keep_alive", () => {
  it("defaults to 15m and is configurable via env", () => {
    assert.equal(loadConfig({}).ollamaKeepAlive, "15m");
    assert.equal(loadConfig({ PIEMBED_OLLAMA_KEEP_ALIVE: "1h" }).ollamaKeepAlive, "1h");
    assert.equal(loadConfig({ OLLAMA_KEEP_ALIVE: "-1" }).ollamaKeepAlive, -1);
  });

  it("is sent with /api/embed and the legacy /api/embeddings fallback", async () => {
    const seen: Array<{ url: string; keep_alive: unknown }> = [];
    const mockFetch: typeof fetch = async (input, init) => {
      const url = String(input);
      seen.push({ url, keep_alive: JSON.parse(String(init?.body)).keep_alive });
      if (url.endsWith("/api/embed")) return new Response("nope", { status: 404 });
      return new Response(JSON.stringify({ embedding: [1, 0, 0] }), { status: 200 });
    };
    const p = new OllamaEmbeddingProvider({ baseUrl: "http://x", defaultModel: "m", fetchFn: mockFetch });
    await p.embed(["a"]);
    assert.deepEqual(seen.map((s) => s.keep_alive), ["15m", "15m"]);
    seen.length = 0;
    const p2 = new OllamaEmbeddingProvider({ baseUrl: "http://x", defaultModel: "m", fetchFn: mockFetch, keepAlive: "1h" });
    await p2.embed(["a"]);
    assert.deepEqual(seen.map((s) => s.keep_alive), ["1h", "1h"]);
  });
});

import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import { describe, it } from "node:test";
import { OllamaEmbeddingProvider } from "../src/providers/ollama-provider.js";
import { EmbeddingOrchestrator } from "../src/providers/orchestrator.js";
import { InProcessTransformersProvider } from "../src/providers/transformers-provider.js";

function makeDeterministicVector(text: string, dim: number): number[] {
  const out = new Array<number>(dim).fill(0);
  const lower = text.toLowerCase();
  for (let i = 0; i < lower.length; i++) {
    const code = lower.charCodeAt(i);
    out[(code + i * 7) % dim] += 1;
  }
  return out;
}

describe("EmbeddingOrchestrator — Remote Ollama & Automatic In-Process Fallback", () => {
  it("uses remote Ollama by default and caches results in LRU", async () => {
    let ollamaCallCount = 0;
    const mockFetch: typeof fetch = async (input, init) => {
      const url = String(input);
      if (url.endsWith("/api/tags")) {
        return new Response(JSON.stringify({ models: [{ name: "bge-m3:latest" }] }), { status: 200 });
      }
      if (url.endsWith("/api/embed")) {
        ollamaCallCount++;
        const body = JSON.parse(String(init?.body)) as { input: string[] };
        const embeddings = body.input.map((t) => makeDeterministicVector(t, 1024));
        return new Response(JSON.stringify({ embeddings }), { status: 200 });
      }
      return new Response("Not Found", { status: 404 });
    };

    const primary = new OllamaEmbeddingProvider({
      baseUrl: "http://192.168.1.50:11434",
      defaultModel: "bge-m3:latest",
      fetchFn: mockFetch,
    });

    const fallback = new InProcessTransformersProvider({
      defaultModel: "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
      cacheDir: path.join(os.tmpdir(), "piembed-test-cache-1"),
      pipelineLoader: async () => async (texts) => {
        const arr = Array.isArray(texts) ? texts : [texts];
        const flat: number[] = [];
        for (const t of arr) flat.push(...makeDeterministicVector(t, 384));
        return { data: new Float32Array(flat) };
      },
    });

    const orchestrator = new EmbeddingOrchestrator({ primary, fallback });

    const res1 = await orchestrator.embed(["Isä meidän", "Herra siunatkoon"]);
    assert.equal(res1.provider, "ollama");
    assert.equal(res1.dimensions, 1024);
    assert.equal(res1.vectors.length, 2);
    assert.equal(ollamaCallCount, 1);

    // Second call with one cached phrase and one new phrase
    const res2 = await orchestrator.embed(["Isä meidän", "Kunnia Isälle"]);
    assert.equal(res2.provider, "ollama");
    assert.equal(res2.cachedHits, 1);
    assert.equal(ollamaCallCount, 2);
  });

  it("automatically trips circuit breaker to in-process fallback when remote Ollama fails", async () => {
    let ollamaOnline = false;
    const mockFetch: typeof fetch = async (input, init) => {
      if (!ollamaOnline) {
        throw new Error("connect ETIMEDOUT 192.168.1.50:11434");
      }
      const url = String(input);
      if (url.endsWith("/api/tags")) {
        return new Response(JSON.stringify({ models: [{ name: "bge-m3:latest" }] }), { status: 200 });
      }
      const body = JSON.parse(String(init?.body)) as { input: string[] };
      const embeddings = body.input.map((t) => makeDeterministicVector(t, 768));
      return new Response(JSON.stringify({ embeddings }), { status: 200 });
    };

    const primary = new OllamaEmbeddingProvider({
      baseUrl: "http://192.168.1.50:11434",
      defaultModel: "nomic-embed-text:latest",
      fetchFn: mockFetch,
    });

    const fallback = new InProcessTransformersProvider({
      defaultModel: "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
      cacheDir: path.join(os.tmpdir(), "piembed-test-cache-2"),
      pipelineLoader: async () => async (texts) => {
        const arr = Array.isArray(texts) ? texts : [texts];
        const flat: number[] = [];
        for (const t of arr) flat.push(...makeDeterministicVector(t, 384));
        return { data: new Float32Array(flat) };
      },
    });

    const orchestrator = new EmbeddingOrchestrator({ primary, fallback });

    // 1. Remote Ollama is offline -> automatically falls back to in-process (384-d)
    const fallbackRes = await orchestrator.embed(["Isä meidän joka olet taivaissa"]);
    assert.equal(fallbackRes.provider, "inprocess");
    assert.equal(fallbackRes.dimensions, 384);
    assert.equal(orchestrator.getStatus().ollama.healthy, false);

    // 2. Remote Ollama comes back online -> health probe recovers it (768-d)
    ollamaOnline = true;
    const recovered = await orchestrator.probeOllamaHealth();
    assert.equal(recovered, true);

    const primaryRes = await orchestrator.embed(["Isä meidän joka olet taivaissa"]);
    assert.equal(primaryRes.provider, "ollama");
    assert.equal(primaryRes.dimensions, 768);
  });
});

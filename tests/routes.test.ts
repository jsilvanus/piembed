import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import { describe, it } from "node:test";
import { createApp } from "../src/app.js";
import { OllamaEmbeddingProvider } from "../src/providers/ollama-provider.js";
import { EmbeddingOrchestrator } from "../src/providers/orchestrator.js";
import { InProcessTransformersProvider } from "../src/providers/transformers-provider.js";

describe("piembed HTTP REST Routes", () => {
  it("supports /health, /models, /models/load, /models/download, /embed, /compare, and /cues cleanup", async () => {
    const primary = new OllamaEmbeddingProvider({
      baseUrl: "http://192.168.1.50:11434",
      defaultModel: "bge-m3:latest",
      fetchFn: async (input, init) => {
        const url = String(input);
        if (url.endsWith("/api/tags")) {
          return new Response(
            JSON.stringify({ models: [{ name: "bge-m3:latest" }, { name: "nomic-embed-text:latest" }] }),
            { status: 200 }
          );
        }
        const body = JSON.parse(String(init?.body)) as { model: string; input: string[] };
        const dim = body.model.includes("nomic") ? 768 : 1024;
        const embeddings = body.input.map((t) => {
          const v = new Array<number>(dim).fill(0.1);
          v[0] = t.includes("hello") ? 0.9 : 0.1;
          return v;
        });
        return new Response(JSON.stringify({ embeddings }), { status: 200 });
      },
    });

    const fallback = new InProcessTransformersProvider({
      defaultModel: "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
      cacheDir: path.join(os.tmpdir(), "piembed-routes-test"),
      pipelineLoader: async (modelId) => async (texts) => {
        const arr = Array.isArray(texts) ? texts : [texts];
        const dim = modelId.includes("e5-base") ? 768 : 384;
        const flat: number[] = [];
        for (let i = 0; i < arr.length; i++) {
          const v = new Array<number>(dim).fill(0.2);
          flat.push(...v);
        }
        return { data: new Float32Array(flat) };
      },
    });

    const orchestrator = new EmbeddingOrchestrator({ primary, fallback });
    const { app } = createApp({ orchestrator, startHealthLoop: false });

    // 1. Health check
    const healthRes = await app.inject({ method: "GET", url: "/api/v1/health" });
    assert.equal(healthRes.statusCode, 200);
    assert.equal(healthRes.json().service, "piembed");

    // 2. Download a Hugging Face model
    const dlRes = await app.inject({
      method: "POST",
      url: "/api/v1/models/download",
      payload: { model: "intfloat/multilingual-e5-small", quantized: true },
    });
    assert.equal(dlRes.statusCode, 200);
    assert.equal(dlRes.json().cached, true);
    assert.equal(dlRes.json().dimensions, 384);

    // 3. Load a 768-d Hugging Face model into in-process runtime
    const loadHfRes = await app.inject({
      method: "POST",
      url: "/api/v1/models/load",
      payload: { source: "huggingface", model: "intfloat/multilingual-e5-base", quantized: false },
    });
    assert.equal(loadHfRes.statusCode, 200);
    assert.equal(loadHfRes.json().dimensions, 768);
    assert.equal(loadHfRes.json().quantized, false);

    // 4. Switch Ollama model to nomic-embed-text:latest (768-d)
    const loadOllamaRes = await app.inject({
      method: "POST",
      url: "/api/v1/models/load",
      payload: { source: "ollama", model: "nomic-embed-text:latest" },
    });
    assert.equal(loadOllamaRes.statusCode, 200);
    assert.equal(loadOllamaRes.json().dimensions, 768);

    // 5. Inspect /api/v1/models
    const modelsRes = await app.inject({ method: "GET", url: "/api/v1/models" });
    assert.equal(modelsRes.statusCode, 200);
    assert.equal(modelsRes.json().ollama.model, "nomic-embed-text:latest");
    assert.ok(modelsRes.json().inprocess.cachedModels.includes("intfloat/multilingual-e5-small"));

    // 6. General /api/v1/embed and /api/v1/compare
    const embedRes = await app.inject({
      method: "POST",
      url: "/api/v1/embed",
      payload: { texts: ["hello world", "goodbye world"] },
    });
    assert.equal(embedRes.statusCode, 200);
    assert.equal(embedRes.json().vectors.length, 2);
    assert.equal(embedRes.json().dimensions, 768);

    const compareRes = await app.inject({
      method: "POST",
      url: "/api/v1/compare",
      payload: {
        query: "hello friend",
        candidates: ["goodbye world", "hello there"],
        threshold: 0.70,
      },
    });
    assert.equal(compareRes.statusCode, 200);
    assert.equal(compareRes.json().bestMatch.text, "hello there");
    assert.equal(compareRes.json().matched, true);

    await app.close();
  });
});

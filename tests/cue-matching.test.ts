import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import { describe, it } from "node:test";
import { createApp } from "../src/app.js";
import { OllamaEmbeddingProvider } from "../src/providers/ollama-provider.js";
import { EmbeddingOrchestrator } from "../src/providers/orchestrator.js";
import { InProcessTransformersProvider } from "../src/providers/transformers-provider.js";

function semanticMockEmbed(text: string, dim: number): number[] {
  const vec = new Array<number>(dim).fill(0.01);
  const lower = text.toLowerCase();
  if (lower.includes("isä meidän") || lower.includes("taivaissa") || lower.includes("pyhitetty")) {
    vec[0] = 0.95;
    vec[1] = 0.85;
  } else if (lower.includes("siunatkoon") || lower.includes("varjelkoon")) {
    vec[2] = 0.95;
    vec[3] = 0.85;
  } else if (lower.includes("uskon") || lower.includes("kaikkivaltiaaseen")) {
    vec[4] = 0.95;
    vec[5] = 0.85;
  } else {
    vec[10] = 0.95;
  }
  return vec;
}

describe("Cue Preflight & Live Matching Engine", () => {
  it("preflights rundown cues on 1024-d Ollama and seamlessly re-indexes to 384-d inprocess if Ollama drops mid-service", async () => {
    let ollamaUp = true;

    const primary = new OllamaEmbeddingProvider({
      baseUrl: "http://192.168.1.50:11434",
      defaultModel: "bge-m3:latest",
      fetchFn: async (input, init) => {
        if (!ollamaUp) {
          throw new Error("Connection refused");
        }
        const url = String(input);
        if (url.endsWith("/api/tags")) {
          return new Response(JSON.stringify({ models: [{ name: "bge-m3:latest" }] }), { status: 200 });
        }
        const body = JSON.parse(String(init?.body)) as { input: string[] };
        const embeddings = body.input.map((t) => semanticMockEmbed(t, 1024));
        return new Response(JSON.stringify({ embeddings }), { status: 200 });
      },
    });

    const fallback = new InProcessTransformersProvider({
      defaultModel: "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
      cacheDir: path.join(os.tmpdir(), "piembed-cue-test"),
      pipelineLoader: async () => async (texts) => {
        const arr = Array.isArray(texts) ? texts : [texts];
        const flat: number[] = [];
        for (const t of arr) flat.push(...semanticMockEmbed(t, 384));
        return { data: new Float32Array(flat) };
      },
    });

    const orchestrator = new EmbeddingOrchestrator({ primary, fallback });
    const { app } = createApp({ orchestrator, startHealthLoop: false });

    // 1. Preflight while Ollama (1024-d) is up
    const preflightRes = await app.inject({
      method: "POST",
      url: "/api/v1/cues/preflight",
      payload: {
        rundownId: "sunday-service",
        cues: [
          { id: "cue_creed", phrase: "Minä uskon Jumalaan Isään Kaikkivaltiaaseen" },
          { id: "cue_paternoster", phrase: "Isä meidän joka olet taivaissa" },
          { id: "cue_benediction", phrase: "Herra siunatkoon teitä ja varjelkoon teitä" },
        ],
      },
    });
    assert.equal(preflightRes.statusCode, 200);
    const preflightData = preflightRes.json();
    assert.equal(preflightData.provider, "ollama");
    assert.equal(preflightData.dimensions, 1024);
    assert.equal(preflightData.registered, 3);

    // 2. Match against Ollama (1024-d)
    const match1 = await app.inject({
      method: "POST",
      url: "/api/v1/cues/match",
      payload: {
        text: "rukoilemme yhdessä Isä meidän joka olet taivaissa pyhitetty olkoon",
        cueIds: ["cue_paternoster", "cue_benediction"],
        threshold: 0.70,
      },
    });
    assert.equal(match1.statusCode, 200);
    const match1Data = match1.json();
    assert.equal(match1Data.matched, true);
    assert.equal(match1Data.bestMatch.id, "cue_paternoster");
    assert.equal(match1Data.provider, "ollama");
    assert.equal(match1Data.dimensions, 1024);

    // 3. Adjacent Ollama computer disconnects mid-service!
    ollamaUp = false;

    // 4. Next caption window arrives -> seamlessly fails over to 384-d inprocess AND re-indexes stale 1024-d cues!
    const match2 = await app.inject({
      method: "POST",
      url: "/api/v1/cues/match",
      payload: {
        text: "Herra siunatkoon teitä ja varjelkoon teitä amen",
        cueIds: ["cue_paternoster", "cue_benediction"],
        threshold: 0.70,
      },
    });
    assert.equal(match2.statusCode, 200);
    const match2Data = match2.json();
    assert.equal(match2Data.matched, true);
    assert.equal(match2Data.bestMatch.id, "cue_benediction");
    assert.equal(match2Data.provider, "inprocess");
    assert.equal(match2Data.dimensions, 384);

    await app.close();
  });
});

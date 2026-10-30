import { createApp } from "./app.js";
import { logger } from "./system/logger.js";

async function main() {
  const { app, config } = createApp();

  try {
    await app.listen({ port: config.port, host: config.host });
    logger.info(
      `piembed listening on http://${config.host}:${config.port} (primary: ${config.ollamaUrl} [${config.ollamaModel}], fallback: ${config.inprocessModel})`
    );
  } catch (err) {
    logger.error("Failed to start piembed server:", err);
    process.exit(1);
  }
}

void main();

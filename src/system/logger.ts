export interface Logger {
  info(msg: string, ...args: unknown[]): void;
  warn(msg: string, ...args: unknown[]): void;
  error(msg: string, ...args: unknown[]): void;
  debug(msg: string, ...args: unknown[]): void;
}

export const logger: Logger = {
  info(msg: string, ...args: unknown[]) {
    console.log(`[piembed] INFO  ${new Date().toISOString()} ${msg}`, ...args);
  },
  warn(msg: string, ...args: unknown[]) {
    console.warn(`[piembed] WARN  ${new Date().toISOString()} ${msg}`, ...args);
  },
  error(msg: string, ...args: unknown[]) {
    console.error(`[piembed] ERROR ${new Date().toISOString()} ${msg}`, ...args);
  },
  debug(msg: string, ...args: unknown[]) {
    if (process.env.DEBUG || process.env.PIEMBED_DEBUG) {
      console.debug(`[piembed] DEBUG ${new Date().toISOString()} ${msg}`, ...args);
    }
  },
};

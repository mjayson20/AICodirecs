/**
 * api.ts — HTTP client for the FastAPI backend.
 */

import * as https from "https";
import * as http from "http";
import { URL } from "url";

export interface CompleteRequest {
  code_context: string;
  max_tokens?: number;
}

export interface CompleteResponse {
  suggestion: string;
  latency_ms: number;
}

export interface HealthResponse {
  status: string;
  model: string;
}

/**
 * Send a POST /complete request to the backend.
 * Returns the suggestion string, or null on error.
 */
export async function fetchCompletion(
  serverUrl: string,
  codeContext: string,
  maxTokens: number,
  timeoutMs = 5000
): Promise<string | null> {
  const body = JSON.stringify({ code_context: codeContext, max_tokens: maxTokens });

  return new Promise((resolve) => {
    try {
      const url = new URL("/complete", serverUrl);
      const isHttps = url.protocol === "https:";
      const lib = isHttps ? https : http;

      const options: http.RequestOptions = {
        hostname: url.hostname,
        port: url.port || (isHttps ? 443 : 80),
        path: url.pathname,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(body),
        },
        timeout: timeoutMs,
      };

      const req = lib.request(options, (res) => {
        let data = "";
        res.on("data", (chunk) => (data += chunk));
        res.on("end", () => {
          try {
            if (res.statusCode === 200) {
              const parsed: CompleteResponse = JSON.parse(data);
              resolve(parsed.suggestion || null);
            } else {
              resolve(null);
            }
          } catch {
            resolve(null);
          }
        });
      });

      req.on("error", () => resolve(null));
      req.on("timeout", () => {
        req.destroy();
        resolve(null);
      });

      req.write(body);
      req.end();
    } catch {
      resolve(null);
    }
  });
}

/**
 * Check if the backend server is reachable.
 */
export async function checkHealth(serverUrl: string): Promise<boolean> {
  return new Promise((resolve) => {
    try {
      const url = new URL("/health", serverUrl);
      const isHttps = url.protocol === "https:";
      const lib = isHttps ? https : http;

      const req = lib.get(
        { hostname: url.hostname, port: url.port || 80, path: "/health", timeout: 2000 },
        (res) => {
          resolve(res.statusCode === 200);
        }
      );
      req.on("error", () => resolve(false));
      req.on("timeout", () => {
        req.destroy();
        resolve(false);
      });
    } catch {
      resolve(false);
    }
  });
}

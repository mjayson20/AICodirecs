/**
 * extension.ts — Entry point for the AI Code Suggester VS Code extension.
 *
 * Features:
 *  - Inline ghost-text suggestions via InlineCompletionItemProvider
 *  - Tab to accept
 *  - Debounced requests to avoid hammering the backend
 *  - Status bar item showing enabled/disabled state
 *  - Commands: triggerSuggestion, showStats, toggleEnabled
 */

import * as vscode from "vscode";
import { checkHealth } from "./api";
import { CodeSuggestionProvider } from "./provider";
import { SuggestionStats } from "./stats";

// ---------------------------------------------------------------------------
// Debounce helper
// ---------------------------------------------------------------------------

function debounce<T extends (...args: Parameters<T>) => void>(
  fn: T,
  delayMs: number
): (...args: Parameters<T>) => void {
  let timer: ReturnType<typeof setTimeout> | undefined;
  return (...args: Parameters<T>) => {
    if (timer) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => fn(...args), delayMs);
  };
}

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  const stats = new SuggestionStats();
  const provider = new CodeSuggestionProvider(stats);

  // ── Status bar ────────────────────────────────────────────────────────────
  const statusBar = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100
  );
  statusBar.command = "aiCodeSuggester.toggleEnabled";
  statusBar.tooltip = "Click to toggle AI Code Suggester";
  context.subscriptions.push(statusBar);

  function updateStatusBar(enabled: boolean, serverOk: boolean): void {
    if (!enabled) {
      statusBar.text = "$(circle-slash) AI Suggest: OFF";
      statusBar.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.warningBackground"
      );
    } else if (!serverOk) {
      statusBar.text = "$(warning) AI Suggest: Server offline";
      statusBar.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.errorBackground"
      );
    } else {
      statusBar.text = "$(sparkle) AI Suggest: ON";
      statusBar.backgroundColor = undefined;
    }
    statusBar.show();
  }

  // ── Health check ──────────────────────────────────────────────────────────
  let serverOnline = false;

  async function pingServer(): Promise<void> {
    const config = vscode.workspace.getConfiguration("aiCodeSuggester");
    const serverUrl = config.get<string>("serverUrl", "http://127.0.0.1:8000");
    const enabled = config.get<boolean>("enabled", true);
    serverOnline = await checkHealth(serverUrl);
    updateStatusBar(enabled, serverOnline);
  }

  // Ping on startup and every 30 seconds
  await pingServer();
  const healthInterval = setInterval(pingServer, 30_000);
  context.subscriptions.push({ dispose: () => clearInterval(healthInterval) });

  // ── Inline completion provider ────────────────────────────────────────────
  const config = vscode.workspace.getConfiguration("aiCodeSuggester");
  const languages = config.get<string[]>("languages", ["python", "javascript", "typescript"]);

  const selector: vscode.DocumentSelector = languages.map((lang) => ({
    language: lang,
    scheme: "file",
  }));

  const registration = vscode.languages.registerInlineCompletionItemProvider(
    selector,
    provider
  );
  context.subscriptions.push(registration);

  // ── Track accepted suggestions ────────────────────────────────────────────
  // VS Code fires onDidChangeTextDocument when ghost text is accepted.
  // We detect acceptance by watching for insertions that match the last suggestion.
  context.subscriptions.push(
    vscode.workspace.onDidChangeTextDocument((event) => {
      const last = provider.getLastSuggestion();
      if (!last) {
        return;
      }
      for (const change of event.contentChanges) {
        if (change.text && last.startsWith(change.text) && change.text.length > 5) {
          stats.recordAccepted();
          break;
        }
      }
    })
  );

  // ── Commands ──────────────────────────────────────────────────────────────

  // Manual trigger
  context.subscriptions.push(
    vscode.commands.registerCommand("aiCodeSuggester.triggerSuggestion", async () => {
      await vscode.commands.executeCommand("editor.action.inlineSuggest.trigger");
    })
  );

  // Show stats
  context.subscriptions.push(
    vscode.commands.registerCommand("aiCodeSuggester.showStats", () => {
      vscode.window.showInformationMessage(stats.summary());
    })
  );

  // Toggle enabled
  context.subscriptions.push(
    vscode.commands.registerCommand("aiCodeSuggester.toggleEnabled", async () => {
      const cfg = vscode.workspace.getConfiguration("aiCodeSuggester");
      const current = cfg.get<boolean>("enabled", true);
      await cfg.update("enabled", !current, vscode.ConfigurationTarget.Global);
      updateStatusBar(!current, serverOnline);
      vscode.window.showInformationMessage(
        `AI Code Suggester ${!current ? "enabled" : "disabled"}`
      );
    })
  );

  // ── Config change listener ────────────────────────────────────────────────
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration(async (e) => {
      if (e.affectsConfiguration("aiCodeSuggester")) {
        await pingServer();
      }
    })
  );

  // ── Notify if server is offline ───────────────────────────────────────────
  if (!serverOnline) {
    const action = await vscode.window.showWarningMessage(
      "AI Code Suggester: Backend server is not running. Start it with `python main.py` in the backend folder.",
      "Dismiss"
    );
    void action; // suppress unused warning
  }

  console.log("AI Code Suggester activated.");
}

export function deactivate(): void {
  console.log("AI Code Suggester deactivated.");
}

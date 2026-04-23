/**
 * provider.ts
 *
 * Implements VS Code's InlineCompletionItemProvider.
 * Captures context around the cursor, calls the backend, and returns
 * ghost-text suggestions that the user can accept with Tab.
 */

import * as vscode from "vscode";
import { fetchCompletion } from "./api";
import { SuggestionStats } from "./stats";

export class CodeSuggestionProvider implements vscode.InlineCompletionItemProvider {
  private _stats: SuggestionStats;
  private _lastSuggestion: string | null = null;

  constructor(stats: SuggestionStats) {
    this._stats = stats;
  }

  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    _context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken
  ): Promise<vscode.InlineCompletionList | null> {
    const config = vscode.workspace.getConfiguration("aiCodeSuggester");

    if (!config.get<boolean>("enabled", true)) {
      return null;
    }

    const serverUrl = config.get<string>("serverUrl", "http://127.0.0.1:8000");
    const maxContextLines = config.get<number>("maxContextLines", 20);
    const maxTokens = config.get<number>("maxTokens", 80);

    // Build context: N lines before cursor + current partial line
    const startLine = Math.max(0, position.line - maxContextLines);
    const contextRange = new vscode.Range(
      new vscode.Position(startLine, 0),
      position
    );
    const codeContext = document.getText(contextRange);

    if (!codeContext.trim()) {
      return null;
    }

    if (token.isCancellationRequested) {
      return null;
    }

    const suggestion = await fetchCompletion(serverUrl, codeContext, maxTokens);

    if (!suggestion || token.isCancellationRequested) {
      return null;
    }

    // Don't show if suggestion is identical to what's already there
    if (suggestion.trim() === codeContext.trim()) {
      return null;
    }

    this._lastSuggestion = suggestion;
    this._stats.recordSuggestion();

    const item = new vscode.InlineCompletionItem(
      suggestion,
      new vscode.Range(position, position)
    );

    return { items: [item] };
  }

  getLastSuggestion(): string | null {
    return this._lastSuggestion;
  }
}

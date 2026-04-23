/**
 * stats.ts — Track suggestion and acceptance counts.
 */

export class SuggestionStats {
  private _totalSuggestions = 0;
  private _acceptedSuggestions = 0;

  recordSuggestion(): void {
    this._totalSuggestions++;
  }

  recordAccepted(): void {
    this._acceptedSuggestions++;
  }

  get total(): number {
    return this._totalSuggestions;
  }

  get accepted(): number {
    return this._acceptedSuggestions;
  }

  get acceptanceRate(): string {
    if (this._totalSuggestions === 0) {
      return "0%";
    }
    const rate = (this._acceptedSuggestions / this._totalSuggestions) * 100;
    return `${rate.toFixed(1)}%`;
  }

  summary(): string {
    return (
      `AI Code Suggester Stats\n` +
      `─────────────────────────\n` +
      `Total suggestions : ${this._totalSuggestions}\n` +
      `Accepted          : ${this._acceptedSuggestions}\n` +
      `Acceptance rate   : ${this.acceptanceRate}`
    );
  }
}

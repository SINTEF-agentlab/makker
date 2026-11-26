// A part of the page that fails to render (a step with a malformed argument,
// say) shows a short note in its place, instead of unmounting the whole app on
// every replay of the session.

import { Component, type ReactNode } from "react";

export class ErrorBoundary extends Component<
  { what: string; children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: unknown) {
    console.error(`Could not show this ${this.props.what}:`, error);
  }

  render() {
    if (this.state.failed) {
      return <div className="sys-note warn">Could not show this {this.props.what}.</div>;
    }
    return this.props.children;
  }
}

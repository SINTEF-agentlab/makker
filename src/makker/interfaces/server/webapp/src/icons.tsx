// Inline SVG icons (no icon dependency). Each is a small presentational component.
import type { ViewKind } from "./store";

const stroke = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

export function PlotIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" {...stroke}>
      <path d="M3 3v18h18" />
      <path d="M7 15l3-4 3 2 4-6" />
    </svg>
  );
}

export function ReportIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" {...stroke}>
      <path d="M6 2h9l5 5v15H6z" />
      <path d="M14 2v6h6M9 13h6M9 17h6" />
    </svg>
  );
}

export function ImageIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" {...stroke}>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <circle cx="9" cy="10" r="1.6" />
      <path d="M21 16l-5-5L5 20" />
    </svg>
  );
}

export function KindIcon({ kind }: { kind: ViewKind }) {
  if (kind === "report") return <ReportIcon />;
  if (kind === "image") return <ImageIcon />;
  return <PlotIcon />;
}

export function SidebarIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" {...stroke}>
      <rect x="3" y="4.5" width="18" height="15" rx="3" />
      <path d="M9.5 4.5v15" />
    </svg>
  );
}

export function PlusIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" {...stroke}>
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}

export function ViewsIcon() {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" {...stroke}>
      <path d="M4 5h16v10H4zM2 19h20" />
    </svg>
  );
}

export function AttachIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" {...stroke}>
      <path d="M21 11.5l-8.5 8.5a5 5 0 01-7-7l8.5-8.5a3.3 3.3 0 014.7 4.7l-8.5 8.5a1.6 1.6 0 01-2.3-2.3l7.8-7.8" />
    </svg>
  );
}

export function ReturnIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" {...stroke}>
      <path d="M9 10l-5 5 5 5" />
      <path d="M20 4v7a4 4 0 0 1-4 4H4" />
    </svg>
  );
}

export function StopIcon() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14">
      <rect x="5" y="5" width="14" height="14" rx="2.5" fill="currentColor" />
    </svg>
  );
}

export function CloseIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" {...stroke}>
      <path d="M6 6l12 12M18 6L6 18" />
    </svg>
  );
}

export function BackIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" {...stroke}>
      <path d="M10 19l-7-7 7-7M3 12h13a5 5 0 015 5v2" />
    </svg>
  );
}

export function PopoutIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" {...stroke}>
      <path d="M14 4h6v6M20 4l-9 9M19 14v5a1 1 0 01-1 1H6a1 1 0 01-1-1V7a1 1 0 011-1h5" />
    </svg>
  );
}

export function ChevronRight() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" {...stroke}>
      <path d="M9 6l6 6-6 6" />
    </svg>
  );
}

/** A figure at its own size inside the panel (the "scale" view mode). */
export function FitIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" {...stroke}>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <rect x="8" y="9" width="8" height="6" />
    </svg>
  );
}

/** A figure reflowed to the whole panel (the "fill" view mode). */
export function FillIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" {...stroke}>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M8 12h8M12 8v8" />
    </svg>
  );
}

/** Rebuild a dead view by re-running its recorded code. */
export function RegenerateIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" {...stroke}>
      <path d="M21 12a9 9 0 11-2.64-6.36" />
      <path d="M21 3v6h-6" />
    </svg>
  );
}

export function ChevronLeft() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" {...stroke}>
      <path d="M15 6l-6 6 6 6" />
    </svg>
  );
}

export function CheckIcon() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" {...stroke}>
      <path d="M5 12l5 5 9-10" />
    </svg>
  );
}

/** Filled when the model is pinned to the top of the model menu. */
export function StarIcon({ filled }: { filled: boolean }) {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" {...stroke} fill={filled ? "currentColor" : "none"}>
      <path d="M12 3l2.7 5.6 6.1.8-4.5 4.2 1.1 6.1L12 16.8 6.6 19.7l1.1-6.1-4.5-4.2 6.1-.8z" />
    </svg>
  );
}

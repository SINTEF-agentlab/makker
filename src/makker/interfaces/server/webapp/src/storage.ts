// Browser storage, for remembering layout and an update across a reload. It can be
// unavailable: a cross-site frame with storage blocked throws on any access, and
// a layout preference is never worth a blank page.

type Area = "local" | "session";

const area = (which: Area): Storage => (which === "local" ? localStorage : sessionStorage);

export function readStored(which: Area, key: string): string | null {
  try {
    return area(which).getItem(key);
  } catch {
    return null;
  }
}

export function writeStored(which: Area, key: string, value: string | null): void {
  try {
    if (value === null) area(which).removeItem(key);
    else area(which).setItem(key, value);
  } catch {
    // Not remembered, which is all that is lost.
  }
}

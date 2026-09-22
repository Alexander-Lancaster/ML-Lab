import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ML-Lab Architecture Viewer",
  description: "A visual inspector for YAML-defined PyTorch architectures.",
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}

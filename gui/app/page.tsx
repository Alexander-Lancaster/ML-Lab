import type { Metadata } from "next";
import { GraphViewer } from "./GraphViewer";

export const metadata: Metadata = {
  title: "ML-Lab Architecture Viewer",
  description: "Inspect neural-network graphs, layer types, connections, and parameters.",
};

export default function Home() {
  return <GraphViewer />;
}

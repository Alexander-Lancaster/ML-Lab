"use client";

import { useMemo, useRef, useState } from "react";
import { load as loadYaml } from "js-yaml";

type Shape = number[];
type InputSpec = { shape: Shape };
type Layer = {
  id: string;
  type: string;
  inputs: string[];
  params?: Record<string, unknown>;
};
type Architecture = {
  format_version: number;
  model: { name: string };
  inputs: Record<string, InputSpec>;
  layers: Layer[];
  outputs: string[];
};
type PositionedNode = {
  id: string;
  kind: "input" | "layer";
  type: string;
  inputs: string[];
  params: Record<string, unknown>;
  shape?: Shape;
  x: number;
  y: number;
};

const NODE_WIDTH = 202;
const NODE_HEIGHT = 116;
const X_GAP = 84;
const Y_GAP = 42;

const samples: Record<string, Architecture> = {
  residual: {
    format_version: 1,
    model: { name: "residual_example" },
    inputs: { image: { shape: [3, 32, 32] } },
    layers: [
      { id: "stem", type: "Conv2d", inputs: ["image"], params: { in_channels: 3, out_channels: 16, kernel_size: 3, padding: 1 } },
      { id: "conv1", type: "Conv2d", inputs: ["stem"], params: { in_channels: 16, out_channels: 16, kernel_size: 3, padding: 1 } },
      { id: "activation", type: "ReLU", inputs: ["conv1"] },
      { id: "conv2", type: "Conv2d", inputs: ["activation"], params: { in_channels: 16, out_channels: 16, kernel_size: 3, padding: 1 } },
      { id: "residual", type: "Add", inputs: ["stem", "conv2"] },
      { id: "output", type: "ReLU", inputs: ["residual"] },
    ],
    outputs: ["output"],
  },
  mnist: {
    format_version: 1,
    model: { name: "mnist_classifier" },
    inputs: { input: { shape: [1, 28, 28] } },
    layers: [
      { id: "layer_0", type: "Conv2d", inputs: ["input"], params: { in_channels: 1, out_channels: 16, kernel_size: 3, padding: 1 } },
      { id: "layer_1", type: "ReLU", inputs: ["layer_0"] },
      { id: "layer_2", type: "MaxPool2d", inputs: ["layer_1"], params: { kernel_size: 2 } },
      { id: "layer_3", type: "Flatten", inputs: ["layer_2"] },
      { id: "layer_4", type: "Linear", inputs: ["layer_3"], params: { in_features: 3136, out_features: 10 } },
    ],
    outputs: ["layer_4"],
  },
};

function normalizeArchitecture(raw: unknown): Architecture {
  if (!raw || typeof raw !== "object") throw new Error("The YAML document must be a mapping.");
  const data = raw as Record<string, unknown>;
  const model = data.model as Record<string, unknown> | undefined;
  if (!model || typeof model.name !== "string") throw new Error("model.name is required.");

  let inputs: Record<string, InputSpec>;
  let legacy = false;
  if (data.inputs && typeof data.inputs === "object") {
    inputs = data.inputs as Record<string, InputSpec>;
  } else if (data.input && typeof data.input === "object") {
    legacy = true;
    inputs = { input: data.input as InputSpec };
  } else {
    throw new Error("The architecture needs input or inputs.");
  }

  if (!Array.isArray(data.layers) || data.layers.length === 0) throw new Error("layers must be a non-empty list.");
  let previous = Object.keys(inputs)[0];
  const layers = data.layers.map((item, index) => {
    if (!item || typeof item !== "object") throw new Error(`layers[${index}] must be a mapping.`);
    const layer = item as Record<string, unknown>;
    if (typeof layer.type !== "string") throw new Error(`layers[${index}].type is required.`);
    const id = typeof layer.id === "string" ? layer.id : legacy ? `layer_${index}` : "";
    if (!id) throw new Error(`layers[${index}].id is required.`);
    const references = Array.isArray(layer.inputs) ? layer.inputs.map(String) : legacy ? [previous] : [];
    if (references.length === 0) throw new Error(`layers[${index}].inputs is required.`);
    previous = id;
    return {
      id,
      type: layer.type,
      inputs: references,
      params: layer.params && typeof layer.params === "object" ? layer.params as Record<string, unknown> : undefined,
    };
  });

  return {
    format_version: typeof data.format_version === "number" ? data.format_version : 1,
    model: { name: model.name },
    inputs,
    layers,
    outputs: Array.isArray(data.outputs) ? data.outputs.map(String) : [layers.at(-1)!.id],
  };
}

function layoutGraph(architecture: Architecture) {
  const level = new Map<string, number>();
  Object.keys(architecture.inputs).forEach((name) => level.set(name, 0));
  architecture.layers.forEach((layer) => {
    const parentLevels = layer.inputs.map((input) => level.get(input) ?? 0);
    level.set(layer.id, Math.max(...parentLevels) + 1);
  });

  const groups = new Map<number, string[]>();
  level.forEach((value, id) => groups.set(value, [...(groups.get(value) ?? []), id]));
  const maxRows = Math.max(...[...groups.values()].map((group) => group.length));
  const canvasHeight = Math.max(420, maxRows * (NODE_HEIGHT + Y_GAP) + 110);
  const layerById = new Map(architecture.layers.map((layer) => [layer.id, layer]));
  const nodes: PositionedNode[] = [];

  [...groups.entries()].sort(([a], [b]) => a - b).forEach(([column, ids]) => {
    const blockHeight = ids.length * NODE_HEIGHT + (ids.length - 1) * Y_GAP;
    const startY = (canvasHeight - blockHeight) / 2;
    ids.forEach((id, row) => {
      const layer = layerById.get(id);
      const input = architecture.inputs[id];
      nodes.push({
        id,
        kind: layer ? "layer" : "input",
        type: layer?.type ?? "Input",
        inputs: layer?.inputs ?? [],
        params: layer?.params ?? {},
        shape: input?.shape,
        x: 48 + column * (NODE_WIDTH + X_GAP),
        y: startY + row * (NODE_HEIGHT + Y_GAP),
      });
    });
  });

  return {
    nodes,
    width: 96 + (Math.max(...level.values()) + 1) * NODE_WIDTH + Math.max(...level.values()) * X_GAP,
    height: canvasHeight,
  };
}

function nodeTone(type: string) {
  if (type === "Input") return "input";
  if (["ReLU", "GELU", "Sigmoid", "Tanh"].includes(type)) return "activation";
  if (["Add", "Concatenate"].includes(type)) return "merge";
  if (type.includes("Pool") || type === "Flatten") return "transform";
  return "layer";
}

function formatValue(value: unknown) {
  if (Array.isArray(value)) return `[${value.join(", ")}]`;
  if (typeof value === "object" && value !== null) return JSON.stringify(value);
  return String(value);
}

export function GraphViewer() {
  const [architecture, setArchitecture] = useState(samples.residual);
  const [sourceLabel, setSourceLabel] = useState("Example · residual.yaml");
  const [selectedId, setSelectedId] = useState("residual");
  const [error, setError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const graph = useMemo(() => layoutGraph(architecture), [architecture]);
  const selected = graph.nodes.find((node) => node.id === selectedId) ?? graph.nodes[0];
  const typeCounts = useMemo(() => {
    const counts = new Map<string, number>();
    architecture.layers.forEach((layer) => counts.set(layer.type, (counts.get(layer.type) ?? 0) + 1));
    return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  }, [architecture]);
  const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
  const outputSet = new Set(architecture.outputs);
  const connectionCount = architecture.layers.reduce((total, layer) => total + layer.inputs.length, 0);

  function selectSample(key: keyof typeof samples) {
    const next = samples[key];
    setArchitecture(next);
    setSourceLabel(`Example · ${key}.yaml`);
    setSelectedId(next.layers[0].id);
    setError("");
  }

  async function loadFile(file: File) {
    try {
      const parsed = normalizeArchitecture(loadYaml(await file.text()));
      setArchitecture(parsed);
      setSourceLabel(`Local file · ${file.name}`);
      setSelectedId(parsed.layers[0].id);
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not read that YAML file.");
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
          <div><div className="product-name">ML-Lab</div><div className="product-mode">Architecture viewer</div></div>
        </div>
        <div className="header-actions">
          <div className="sample-switcher" aria-label="Example architectures">
            <button className={sourceLabel.includes("residual") ? "active" : ""} onClick={() => selectSample("residual")}>Residual</button>
            <button className={sourceLabel.includes("mnist") ? "active" : ""} onClick={() => selectSample("mnist")}>MNIST</button>
          </div>
          <input ref={fileRef} className="visually-hidden" type="file" accept=".yaml,.yml,text/yaml" onChange={(event) => event.target.files?.[0] && loadFile(event.target.files[0])} />
          <button className="open-button" onClick={() => fileRef.current?.click()}><span aria-hidden="true">↑</span> Open YAML</button>
        </div>
      </header>

      <section className="model-heading">
        <div>
          <div className="eyebrow">{sourceLabel}</div>
          <h1>{architecture.model.name}</h1>
          <p>A read-only view of the computation graph and its configured layers.</p>
        </div>
        <div className="status-pill"><span /> Parsed successfully</div>
      </section>

      {error && <div className="error-banner" role="alert"><strong>Couldn’t open file.</strong> {error}</div>}

      <section className="stats-row" aria-label="Architecture summary">
        <article><span>Graph nodes</span><strong>{architecture.layers.length + Object.keys(architecture.inputs).length}</strong><small>{Object.keys(architecture.inputs).length} input + {architecture.layers.length} layers</small></article>
        <article><span>Layers</span><strong>{architecture.layers.length}</strong><small>{typeCounts.length} distinct types</small></article>
        <article><span>Connections</span><strong>{connectionCount}</strong><small>{connectionCount > architecture.layers.length ? "Includes branch paths" : "Sequential flow"}</small></article>
        <article><span>Outputs</span><strong>{architecture.outputs.length}</strong><small>{architecture.outputs.join(", ")}</small></article>
      </section>

      <div className="workspace">
        <aside className="left-panel panel">
          <div className="panel-heading"><span>Layer inventory</span><span className="count-badge">{architecture.layers.length}</span></div>
          <div className="type-list">
            {typeCounts.map(([type, count]) => <div className="type-row" key={type}><span className={`type-dot ${nodeTone(type)}`} /><span>{type}</span><strong>{count}</strong></div>)}
          </div>
          <div className="legend">
            <div className="panel-heading">Legend</div>
            <div><span className="legend-swatch input" />Input</div>
            <div><span className="legend-swatch layer" />Parameterized layer</div>
            <div><span className="legend-swatch activation" />Activation</div>
            <div><span className="legend-swatch merge" />Merge operation</div>
          </div>
        </aside>

        <section className="graph-panel panel" aria-label="Architecture graph">
          <div className="canvas-toolbar"><div><span className="live-dot" />Graph</div><span>Click a node to inspect it</span></div>
          <div className="canvas-scroll">
            <div className="graph-canvas" style={{ width: graph.width, height: graph.height }}>
              <div className="grid-texture" />
              <svg className="edges" width={graph.width} height={graph.height} aria-hidden="true">
                {graph.nodes.flatMap((node) => node.inputs.map((sourceId) => {
                  const source = nodeById.get(sourceId);
                  if (!source) return null;
                  const x1 = source.x + NODE_WIDTH;
                  const y1 = source.y + NODE_HEIGHT / 2;
                  const x2 = node.x;
                  const y2 = node.y + NODE_HEIGHT / 2;
                  const bend = Math.max(42, (x2 - x1) * 0.48);
                  return <path key={`${sourceId}-${node.id}`} d={`M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`} />;
                }))}
              </svg>
              {graph.nodes.map((node) => (
                <button
                  key={node.id}
                  className={`graph-node ${nodeTone(node.type)} ${selected?.id === node.id ? "selected" : ""}`}
                  style={{ left: node.x, top: node.y, width: NODE_WIDTH, height: NODE_HEIGHT }}
                  onClick={() => setSelectedId(node.id)}
                >
                  <div className="node-topline"><span>{node.kind === "input" ? "SOURCE" : node.type.toUpperCase()}</span>{outputSet.has(node.id) && <em>OUTPUT</em>}</div>
                  <strong>{node.id}</strong>
                  <div className="node-meta">{node.shape ? `shape [${node.shape.join(", ")}]` : `${Object.keys(node.params).length} parameters`}</div>
                  <i className="port input-port" /><i className="port output-port" />
                </button>
              ))}
            </div>
          </div>
        </section>

        <aside className="right-panel panel">
          <div className="panel-heading">Inspector <span className="readonly-tag">READ ONLY</span></div>
          {selected && <>
            <div className={`inspector-icon ${nodeTone(selected.type)}`}>{selected.type.slice(0, 2).toUpperCase()}</div>
            <div className="inspector-title"><h2>{selected.id}</h2><span>{selected.type}</span></div>
            <dl className="detail-list">
              <div><dt>Node kind</dt><dd>{selected.kind}</dd></div>
              <div><dt>Inputs</dt><dd>{selected.inputs.length ? selected.inputs.join(", ") : "—"}</dd></div>
              <div><dt>Model output</dt><dd>{outputSet.has(selected.id) ? "Yes" : "No"}</dd></div>
              {selected.shape && <div><dt>Declared shape</dt><dd>[{selected.shape.join(", ")}]</dd></div>}
            </dl>
            <div className="parameter-heading">Parameters <span>{Object.keys(selected.params).length}</span></div>
            <div className="parameter-list">
              {Object.entries(selected.params).map(([key, value]) => <div key={key}><span>{key}</span><code>{formatValue(value)}</code></div>)}
              {Object.keys(selected.params).length === 0 && <p>No configured parameters.</p>}
            </div>
          </>}
        </aside>
      </div>

      <footer><span>Format v{architecture.format_version}</span><span>•</span><span>{architecture.layers.length} layers inspected</span><span className="footer-spacer" /><span>Local files stay in your browser</span></footer>
    </main>
  );
}

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
const X_GAP = 62;
const Y_GAP = 76;

const layerTypes = [
  "Conv1d", "Conv2d", "Conv3d", "Linear", "LazyLinear",
  "ReLU", "GELU", "LeakyReLU", "ELU", "Sigmoid", "Tanh",
  "MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d",
  "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
  "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm",
  "Dropout", "Dropout2d", "Dropout3d", "Flatten", "Identity",
  "Softmax", "LogSoftmax", "Add", "Concatenate",
];

const defaultParams: Record<string, Record<string, unknown>> = {
  Conv1d: { in_channels: 1, out_channels: 16, kernel_size: 3 },
  Conv2d: { in_channels: 1, out_channels: 16, kernel_size: 3, padding: 1 },
  Conv3d: { in_channels: 1, out_channels: 16, kernel_size: 3 },
  Linear: { in_features: 128, out_features: 10 },
  LazyLinear: { out_features: 10 },
  LeakyReLU: { negative_slope: 0.01 },
  MaxPool1d: { kernel_size: 2 }, MaxPool2d: { kernel_size: 2 }, MaxPool3d: { kernel_size: 2 },
  AvgPool1d: { kernel_size: 2 }, AvgPool2d: { kernel_size: 2 }, AvgPool3d: { kernel_size: 2 },
  AdaptiveAvgPool1d: { output_size: 1 }, AdaptiveAvgPool2d: { output_size: [1, 1] }, AdaptiveAvgPool3d: { output_size: [1, 1, 1] },
  BatchNorm1d: { num_features: 16 }, BatchNorm2d: { num_features: 16 }, BatchNorm3d: { num_features: 16 },
  LayerNorm: { normalized_shape: 16 }, Dropout: { p: 0.5 }, Dropout2d: { p: 0.5 }, Dropout3d: { p: 0.5 },
  Softmax: { dim: 1 }, LogSoftmax: { dim: 1 }, Concatenate: { dim: 1 },
};

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
  const maxColumns = Math.max(...[...groups.values()].map((group) => group.length));
  const maxLevel = Math.max(...level.values());
  const canvasWidth = Math.max(620, maxColumns * NODE_WIDTH + (maxColumns - 1) * X_GAP + 112);
  const canvasHeight = 96 + (maxLevel + 1) * NODE_HEIGHT + maxLevel * Y_GAP;
  const layerById = new Map(architecture.layers.map((layer) => [layer.id, layer]));
  const nodes: PositionedNode[] = [];

  [...groups.entries()].sort(([a], [b]) => a - b).forEach(([row, ids]) => {
    const blockWidth = ids.length * NODE_WIDTH + (ids.length - 1) * X_GAP;
    const startX = (canvasWidth - blockWidth) / 2;
    ids.forEach((id, column) => {
      const layer = layerById.get(id);
      const input = architecture.inputs[id];
      nodes.push({
        id,
        kind: layer ? "layer" : "input",
        type: layer?.type ?? "Input",
        inputs: layer?.inputs ?? [],
        params: layer?.params ?? {},
        shape: input?.shape,
        x: startX + column * (NODE_WIDTH + X_GAP),
        y: 48 + row * (NODE_HEIGHT + Y_GAP),
      });
    });
  });

  return {
    nodes,
    width: canvasWidth,
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

function parseEditorValue(value: string): unknown {
  const trimmed = value.trim();
  if (!trimmed) return "";
  try { return JSON.parse(trimmed); } catch { return value; }
}

function cloneArchitecture(architecture: Architecture): Architecture {
  return structuredClone(architecture);
}

function graphIssues(architecture: Architecture): string[] {
  const known = new Set(Object.keys(architecture.inputs));
  const issues: string[] = [];
  architecture.layers.forEach((layer) => {
    layer.inputs.forEach((input) => {
      if (!known.has(input)) issues.push(`${layer.id} references missing node “${input}”.`);
    });
    const isMerge = layer.type === "Add" || layer.type === "Concatenate";
    if (isMerge && layer.inputs.length < 2) issues.push(`${layer.type} requires at least two inputs.`);
    if (!isMerge && layer.inputs.length !== 1) issues.push(`${layer.type} expects one input; ${layer.id} has ${layer.inputs.length}.`);
    known.add(layer.id);
  });
  return issues;
}

export function GraphViewer() {
  const [architecture, setArchitecture] = useState(() => cloneArchitecture(samples.residual));
  const [baseline, setBaseline] = useState(() => cloneArchitecture(samples.residual));
  const [sourceLabel, setSourceLabel] = useState("Example · residual.yaml");
  const [selectedId, setSelectedId] = useState("residual");
  const [error, setError] = useState("");
  const [hasEdits, setHasEdits] = useState(false);
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
  const issues = graphIssues(architecture);

  function selectSample(key: keyof typeof samples) {
    const next = cloneArchitecture(samples[key]);
    setArchitecture(next);
    setBaseline(cloneArchitecture(next));
    setSourceLabel(`Example · ${key}.yaml`);
    setSelectedId(next.layers[0].id);
    setError("");
    setHasEdits(false);
  }

  async function loadFile(file: File) {
    try {
      const parsed = normalizeArchitecture(loadYaml(await file.text()));
      setArchitecture(parsed);
      setBaseline(cloneArchitecture(parsed));
      setSourceLabel(`Local file · ${file.name}`);
      setSelectedId(parsed.layers[0].id);
      setError("");
      setHasEdits(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not read that YAML file.");
    }
  }

  function updateSelectedLayer(update: (layer: Layer) => Layer) {
    if (!selected || selected.kind !== "layer") return;
    setArchitecture((current) => ({
      ...current,
      layers: current.layers.map((layer) => layer.id === selected.id ? update(layer) : layer),
    }));
    setHasEdits(true);
  }

  function resetEdits() {
    setArchitecture(cloneArchitecture(baseline));
    setHasEdits(false);
    setError("");
  }

  function renameParameter(oldKey: string, newKey: string) {
    if (!newKey || newKey === oldKey) return;
    updateSelectedLayer((layer) => {
      const entries = Object.entries(layer.params ?? {}).map(([key, value]) => key === oldKey ? [newKey, value] : [key, value]);
      return { ...layer, params: Object.fromEntries(entries) };
    });
  }

  function updateParameter(key: string, value: string) {
    updateSelectedLayer((layer) => ({ ...layer, params: { ...(layer.params ?? {}), [key]: parseEditorValue(value) } }));
  }

  function removeParameter(key: string) {
    updateSelectedLayer((layer) => {
      const params = { ...(layer.params ?? {}) };
      delete params[key];
      return { ...layer, params };
    });
  }

  function addParameter() {
    updateSelectedLayer((layer) => {
      const params = { ...(layer.params ?? {}) };
      let index = 1;
      while (`parameter_${index}` in params) index += 1;
      params[`parameter_${index}`] = 0;
      return { ...layer, params };
    });
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
          {hasEdits && <button className="reset-button" onClick={resetEdits}>Reset changes</button>}
          <input ref={fileRef} className="visually-hidden" type="file" accept=".yaml,.yml,text/yaml" onChange={(event) => event.target.files?.[0] && loadFile(event.target.files[0])} />
          <button className="open-button" onClick={() => fileRef.current?.click()}><span aria-hidden="true">↑</span> Open YAML</button>
        </div>
      </header>

      <section className="model-heading">
        <div>
          <div className="eyebrow">{sourceLabel}</div>
          <h1>{architecture.model.name}</h1>
          <p>Select a layer to change its type and configured parameters.</p>
        </div>
        <div className={`status-pill ${issues.length ? "warning" : ""}`}><span />{issues.length ? `${issues.length} issue${issues.length === 1 ? "" : "s"}` : hasEdits ? "Browser edits valid" : "Parsed successfully"}</div>
      </section>

      {error && <div className="error-banner" role="alert"><strong>Couldn’t open file.</strong> {error}</div>}
      {issues.length > 0 && <div className="issue-banner" role="status"><strong>Needs attention.</strong> {issues[0]}</div>}

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
          <div className="canvas-toolbar"><div><span className="live-dot" />Graph · top to bottom</div><span>Click a layer to edit it</span></div>
          <div className="canvas-scroll">
            <div className="graph-canvas" style={{ width: graph.width, height: graph.height }}>
              <div className="grid-texture" />
              <svg className="edges" width={graph.width} height={graph.height} aria-hidden="true">
                {graph.nodes.flatMap((node) => node.inputs.map((sourceId) => {
                  const source = nodeById.get(sourceId);
                  if (!source) return null;
                  const x1 = source.x + NODE_WIDTH / 2;
                  const y1 = source.y + NODE_HEIGHT;
                  const x2 = node.x + NODE_WIDTH / 2;
                  const y2 = node.y;
                  const bend = Math.max(38, (y2 - y1) * 0.5);
                  return <path key={`${sourceId}-${node.id}`} d={`M ${x1} ${y1} C ${x1} ${y1 + bend}, ${x2} ${y2 - bend}, ${x2} ${y2}`} />;
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
          <div className="panel-heading">Node editor <span className="editor-tag">LIVE</span></div>
          {selected && <>
            <div className={`inspector-icon ${nodeTone(selected.type)}`}>{selected.type.slice(0, 2).toUpperCase()}</div>
            <div className="inspector-title"><h2>{selected.id}</h2><span>{selected.type}</span></div>
            <dl className="detail-list">
              <div><dt>Node kind</dt><dd>{selected.kind}</dd></div>
              <div><dt>Inputs</dt><dd>{selected.inputs.length ? selected.inputs.join(", ") : "—"}</dd></div>
              <div><dt>Model output</dt><dd>{outputSet.has(selected.id) ? "Yes" : "No"}</dd></div>
              {selected.shape && <div><dt>Declared shape</dt><dd>[{selected.shape.join(", ")}]</dd></div>}
            </dl>
            {selected.kind === "layer" ? <>
              <div className="field-section">
                <label htmlFor="layer-type">Layer type</label>
                <select
                  id="layer-type"
                  value={selected.type}
                  onChange={(event) => updateSelectedLayer((layer) => ({ ...layer, type: event.target.value, params: { ...(defaultParams[event.target.value] ?? {}) } }))}
                >
                  {layerTypes.map((type) => <option key={type}>{type}</option>)}
                </select>
                <small>Changing type replaces parameters with sensible defaults.</small>
              </div>
              <div className="parameter-heading">Parameters <span>{Object.keys(selected.params).length}</span></div>
              <div className="parameter-editor">
                {Object.entries(selected.params).map(([key, value]) => <div className="parameter-edit-row" key={`${selected.id}-${key}`}>
                  <input aria-label="Parameter name" defaultValue={key} onBlur={(event) => renameParameter(key, event.target.value.trim())} />
                  <input aria-label={`${key} value`} defaultValue={formatValue(value)} onBlur={(event) => updateParameter(key, event.target.value)} />
                  <button aria-label={`Remove ${key}`} title={`Remove ${key}`} onClick={() => removeParameter(key)}>×</button>
                </div>)}
                {Object.keys(selected.params).length === 0 && <p>No configured parameters. This layer may not require any.</p>}
                <button className="add-parameter" onClick={addParameter}>+ Add parameter</button>
              </div>
            </> : <div className="input-notice"><strong>Model input</strong><p>Input editing is not part of this layer-editing milestone.</p></div>}
          </>}
        </aside>
      </div>

      <footer><span>Format v{architecture.format_version}</span><span>•</span><span>{architecture.layers.length} layers</span>{hasEdits && <><span>•</span><span>Unsaved browser edits</span></>}<span className="footer-spacer" /><span>Local files stay in your browser</span></footer>
    </main>
  );
}

import assert from "node:assert/strict";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the ML-Lab architecture viewer", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /<title>ML-Lab Architecture Viewer<\/title>/i);
  assert.match(html, /Architecture viewer/i);
  assert.match(html, /residual_example/i);
  assert.match(html, /Graph nodes/i);
  assert.match(html, /Layer inventory/i);
  assert.match(html, /Node editor/i);
  assert.match(html, /Save YAML/i);
  assert.match(html, /Vertical/i);
  assert.match(html, /Horizontal/i);
  assert.match(html, /Layer type/i);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});

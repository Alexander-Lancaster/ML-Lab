import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { fileURLToPath } from "node:url";
import Ajv2020 from "ajv/dist/2020.js";

const schemaUrl = new URL("../../schema/architecture.schema.json", import.meta.url);
const fixturesUrl = new URL("../../schema/validation-fixtures.json", import.meta.url);
const schema = JSON.parse(await readFile(fileURLToPath(schemaUrl), "utf8"));
const fixtures = JSON.parse(await readFile(fileURLToPath(fixturesUrl), "utf8"));
const validate = new Ajv2020({ allErrors: true, strict: true }).compile(schema);

for (const fixture of fixtures) {
  test(`shared schema: ${fixture.name}`, () => {
    const actual = validate(fixture.document);
    assert.equal(actual, fixture.valid, JSON.stringify(validate.errors, null, 2));
  });
}

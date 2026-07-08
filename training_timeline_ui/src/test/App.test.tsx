import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import App from "../App";

afterEach(() => {
  vi.unstubAllGlobals();
});

test("renders the timeline dashboard shell", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      json: async () => ({ runs: [] }),
    })),
  );

  render(<App />);

  expect(await screen.findByText("Training Timeline")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Sources" })).toBeInTheDocument();
});

test("shows a visible backend error instead of a false empty state", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: false,
      status: 404,
      json: async () => ({}),
    })),
  );

  render(<App />);

  expect(await screen.findByRole("alert")).toHaveTextContent("Could not load runs");
});

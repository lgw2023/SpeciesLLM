import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import App from "../App";

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

import { cleanDisplayText } from "../display";

test("cleanDisplayText removes markdown syntax and truncated link targets", () => {
  expect(cleanDisplayText("> **See** [5/18 smoke](../training_output_500m_stab_")).toBe("See 5/18 smoke");
});

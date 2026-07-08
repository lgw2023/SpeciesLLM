import { Save } from "lucide-react";
import { useState } from "react";
import type { AnalysisNote, DiagnosticEvent } from "../types";

type AnalysisPanelProps = {
  runId: string;
  diagnostics: DiagnosticEvent[];
  notes: AnalysisNote[];
  onCreate: (payload: Partial<AnalysisNote>) => void | Promise<void>;
  onUpdate: (noteId: string, payload: Partial<AnalysisNote>) => void | Promise<void>;
};

export function AnalysisPanel({ diagnostics, notes, onCreate }: AnalysisPanelProps) {
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [confidence, setConfidence] = useState("medium");

  async function submit() {
    if (!title.trim() || !body.trim()) {
      return;
    }
    await onCreate({
      note_type: "deep_review",
      title,
      body,
      confidence,
      supersedes_diagnostic_ids: diagnostics.flatMap((event) => (event.id ? [event.id] : [])),
      evidence_refs: [],
      author: "local",
    });
    setTitle("");
    setBody("");
    setConfidence("medium");
  }

  return (
    <section className="analysis-panel">
      <div className="subsection-heading">
        <h2>Deep review</h2>
      </div>
      <div className="note-list">
        {notes.map((note) => (
          <article className="note-card" key={note.id}>
            <div className="note-card-header">
              <h3>{note.title}</h3>
              <span>{note.confidence} confidence</span>
            </div>
            <p className="note-body">{note.body}</p>
            {(note.evidence_refs ?? []).length ? (
              <div className="tag-row">
                {(note.evidence_refs ?? []).map((ref) => (
                  <span key={ref}>{ref}</span>
                ))}
              </div>
            ) : null}
          </article>
        ))}
      </div>
      <div className="analysis-form">
        <input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Title" />
        <textarea value={body} onChange={(event) => setBody(event.target.value)} placeholder="Review note" />
        <select value={confidence} onChange={(event) => setConfidence(event.target.value)}>
          <option value="low">low</option>
          <option value="medium">medium</option>
          <option value="high">high</option>
        </select>
        <button className="icon-button text-button" type="button" onClick={submit}>
          <Save aria-hidden="true" size={16} />
          Save
        </button>
      </div>
    </section>
  );
}

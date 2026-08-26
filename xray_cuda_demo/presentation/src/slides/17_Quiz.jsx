import { useState } from "react";

const QUESTIONS = [
  {
    q: "Why can a GPU be faster for this workload?",
    options: [
      "It always has a faster CPU",
      "It can execute many compatible operations in parallel",
      "It removes all memory transfers",
      "CUDA automatically optimizes every algorithm",
    ],
    correct: 1,
    explain: "Filters like Gaussian or Median compute each output pixel largely independently, which maps well onto many parallel GPU threads.",
  },
  {
    q: "What is a CUDA thread, compared to a CPU thread?",
    options: [
      "The exact same concept, just on different hardware",
      "A much heavier-weight execution context than a CPU thread",
      "A lightweight execution unit; thousands can be resident on the GPU at once",
      "A term for a whole GPU core",
    ],
    correct: 2,
    explain: "CUDA threads are far lighter-weight than CPU threads, and GPU hardware schedules them in groups called warps.",
  },
  {
    q: "What actually changed between Basic CUDA and Enhanced CUDA?",
    options: [
      "The filter math was changed to a different algorithm",
      "The same math, but with memory access patterns and kernel specialization optimized per filter",
      "Enhanced CUDA runs on the CPU instead",
      "Nothing measurable changed",
    ],
    correct: 1,
    explain: "Enhanced CUDA keeps the same filter results (verified bit-exact against Basic) — it's the same computation, executed more efficiently.",
  },
  {
    q: "Why does batch size affect throughput?",
    options: [
      "It doesn't — batch size has no effect",
      "Processing more images per GPU call amortizes fixed per-call overhead",
      "Larger batches always slow the GPU down",
      "Batch size only matters for the CPU",
    ],
    correct: 1,
    explain: "Each GPU call carries some fixed overhead; spreading that over more images per call generally improves images/second, up to a point.",
  },
  {
    q: "Why was the persistent-GPU-buffer optimization rejected?",
    options: [
      "It produced incorrect results",
      "It was slower in every single test",
      "It won a same-shape repeated-call microbenchmark but regressed ~28% under a realistic, varied workload",
      "It required a different GPU",
    ],
    correct: 2,
    explain: "The microbenchmark and the realistic workload disagreed — and the realistic workload is the one that matters for a real application.",
  },
];

export function Quiz() {
  const [answers, setAnswers] = useState({});

  function choose(qi, oi) {
    setAnswers((prev) => ({ ...prev, [qi]: oi }));
  }

  const answeredCount = Object.keys(answers).length;
  const correctCount = QUESTIONS.reduce((acc, q, i) => acc + (answers[i] === q.correct ? 1 : 0), 0);

  return (
    <div className="slide">
      <h1>Quick Quiz</h1>
      <p className="subtitle">
        {answeredCount === QUESTIONS.length
          ? `You got ${correctCount} of ${QUESTIONS.length} correct.`
          : "Optional — test what you've picked up."}
      </p>

      {QUESTIONS.map((q, qi) => (
        <div className="card" key={qi} style={{ marginBottom: "1rem" }}>
          <p style={{ fontWeight: 600 }}>{qi + 1}. {q.q}</p>
          {q.options.map((opt, oi) => {
            const chosen = answers[qi];
            let cls = "quiz-option";
            if (chosen !== undefined) {
              if (oi === q.correct) cls += " correct";
              else if (oi === chosen) cls += " incorrect";
            }
            return (
              <button key={oi} className={cls} onClick={() => choose(qi, oi)} disabled={chosen !== undefined}>
                {opt}
              </button>
            );
          })}
          {answers[qi] !== undefined && (
            <p style={{ color: "var(--muted)", fontSize: "0.85rem", marginTop: "0.5rem" }}>{q.explain}</p>
          )}
        </div>
      ))}
    </div>
  );
}

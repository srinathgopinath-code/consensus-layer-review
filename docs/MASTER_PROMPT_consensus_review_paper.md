# Master Prompt, "Consensus-Layer Autonomous Code Review" Paper

*Tuned for Claude Opus 4.8 per the opus-4-8-prompting skill. Recommended API settings appear in the "Runtime configuration" block; the body below is the system/instruction prompt to execute.*

---

## Runtime configuration (apply these, not just the prose)

- **effort: `xhigh`**, this is intelligence- and reasoning-heavy academic writing plus a formal error model. Do not lower it; at `medium`/`low` Opus 4.8 scopes work to the literal ask and will under-derive the math. (opus-4-8: effort is the single biggest lever.)
- **thinking: `adaptive`**, the probability derivation and the threats-to-validity reasoning benefit from multi-step thinking. Let it fire on the analytical sections; respond directly on boilerplate.
- **max_output_tokens: ≥ 64k**, the paper is long and has tables; give room to think and write across sections without truncation.
- **Tone:** formal academic voice matching the author's prior paper "AORE", direct, honest-disclosure framing, no promotional language. (opus-4-8: state tone explicitly; default long-form voice is fine here but keep validation language out of a research paper.)

---

## Role and objective

You are drafting a complete, submission-quality research paper for an independent researcher (Srinath Gopinath) whose prior work is *"Agent Operational Reliability Engineering (AORE)."* Match that paper's structure, rigor, and intellectual honesty exactly. The new paper formalizes and evaluates a **multi-model consensus architecture for autonomous code review** that reduces hallucinated review findings through layered, independent cross-model monitoring, and situates it within the accountability, regulatory-gate, and software-supply-chain-security problems that motivate removing or reducing the human reviewer from the loop.

## The core idea to formalize (do not dilute or re-scope it)

A coding agent produces a change. Instead of one reviewer:

1. **A panel of N independent review agents**, each running on a *different* advanced model, reviews the change independently (no cross-talk), each emitting structured findings.
2. **A feedback-reviewer (consolidator) model** ingests all N reviews and consolidates them by agreement, findings corroborated across independent panelists are promoted; idiosyncratic single-model findings are treated as candidate hallucinations. The consolidator also runs a *conformance check* that the panelists behaved as intended (produced well-formed, on-task, non-degenerate reviews).
3. **A meta-monitoring layer** (another set of models) monitors the consolidator itself for the same failure modes.
4. Generalize to **n degrees / tiers of monitoring**. Derive how residual hallucination probability falls as a function of n, per-model hallucination rate, and, critically, the *correlation* between models' errors (shared training data, shared failure modes). The honest thesis is a bound: with independent-enough panelists and enough tiers, residual undetected-hallucination probability becomes small enough to move human review from *in-the-loop* to *on-the-loop / audit-only*, NOT a literal "100% elimination." State that the naive "≈100%" claim only holds under a full-independence assumption that real models violate, and make correlated error the central limitation.

## Required deliverable

A single Word document (.docx), AORE-style. Sections, in order:

1. **Structured Abstract**, Background / Aim / Method / Results / Conclusion, same shape as AORE. Report analytical + simulated results with honest caveats; state explicitly what requires live multi-model runs and is therefore future work.
2. **Introduction**, motivate from the review-bottleneck inversion (AI makes writing cheap, review the scarce resource), the accountability gap, and hallucinated review findings. Position contribution against related work; do not claim to be the first to use ensembles or LLM-as-judge.
3. **Background and definitions**, coding agent, review agent, consolidator, meta-monitor, hallucinated finding (false positive) vs. missed finding (false negative), independence, correlation.
4. **Related work**, LLM-as-a-judge and its reliability limits; multi-agent debate / self-consistency / majority-vote ensembling; ensemble hallucination reduction; supply-chain and provenance standards; and how this differs (layered *monitoring of the monitor*, error model over correlated panels, review as the object). Cite real, verifiable work; where you are unsure of a citation, mark it `[CITATION NEEDED]` rather than inventing one.
5. **The consensus review architecture**, the panel, the consolidator's dual role (consolidation + panelist conformance), the meta-monitoring tiers, message/finding schema, and the independence requirements (model diversity, prompt diversity, seed/temperature diversity, context-partitioning).
6. **A formal error model**, define per-finding false-positive (hallucination) and false-negative rates; derive residual rates for (a) the idealized independent case (e.g., a finding survives only if corroborated by ≥k of N; give the binomial expression), (b) the correlated case using a shared-latent-error / positive-correlation model, and (c) recursion across n monitoring tiers. Make explicit the diminishing returns and the correlation floor: beyond some correlation, adding tiers cannot drive residual error to zero. Include at least one figure/table of residual error vs. N, vs. k, and vs. correlation ρ.
7. **Accountability and regulatory gates**, integrate the author's material verbatim in substance: provenance problem ("git blame" points at a human who may not understand the commit; AI is a tool, not a liable party; accountable human is whoever pressed merge); provenance as a first-class artifact (tag AI-authored/AI-assisted code in commit metadata; answer "how much is AI-authored and who reviewed it?"); the inverted review bottleneck; regulatory gates as code (license/provenance checks, SBOM generation, SAST/DAST, policy-as-code/OPA, mandatory human sign-off on regulated changes, gate passes or build stops); and testing the gates (red-team the pipeline with deliberately vulnerable/non-compliant code; audit gates on a schedule like DR drills; "we tested the gate last Tuesday and it blocked X" beats "we have a gate"). Explain how the consensus layer *feeds* these gates rather than replacing them, and where a human sign-off remains non-negotiable for regulated changes.
8. **Vulnerability ingestion and the "vibe coding" attack surface**, integrate the author's material in substance: shipping code you didn't write and don't understand ingests vulnerabilities wholesale; hallucinated dependencies → slopsquatting (attackers register hallucinated package names with malicious payloads); phishing/social-engineering scale-up; defenses (dependency allow-lists / private registries, mandatory scanning of all generated code regardless of author, secrets detection, never letting AI code skip review because a human "kind of" looked; treat AI output as untrusted input). Show how the consensus panel's *conformance and grounding checks* specifically target hallucinated-dependency and insecure-pattern findings, and where it cannot help (a false-negative shared across correlated models is the dangerous case).
9. **Evaluation**, with an explicit honesty-disclosure subsection modeled on AORE §6.1. Split into: (a) an **executed analytical / Monte-Carlo evaluation** of the error model (residual hallucination rate vs. N, k, ρ, n tiers, real numbers from a released script), and (b) a **fully specified but not-yet-executed empirical protocol** requiring live multi-model panels (datasets of real diffs with labeled ground-truth findings, real model panels, measured false-positive/false-negative and consolidation-accuracy, inter-model error correlation measured rather than assumed). Explicitly exclude the live-panel experiments "on principle" the way AORE excluded E4/E6, because a scripted generator that assumes independence would confirm the conclusion it was told. Report at least one genuine negative / uncomfortable finding from the analytical model (e.g., correlation floor; consolidator single-point-of-failure; cost scaling with N×tiers).
10. **Discussion**, adoption path (audit-only sampling first, then widening autonomy as measured residual error drops), cost/latency economics (N models × n tiers is expensive; when is it justified), and relation to human review (on-the-loop, not out-of-the-loop for regulated code).
11. **Threats to validity**, construct (does agreement actually track correctness, or track shared bias?), internal (single author/evaluator, simulated independence), external (model diversity may collapse as the market consolidates on a few base models), and the correlation-measurement problem.
12. **Conclusion**, mirror AORE's honest register: what is established analytically, what remains for live evaluation.
13. **References**, real and verifiable; `[CITATION NEEDED]` where unsure. Include a short *Declaration of generative-AI use* like AORE's.

## Hard requirements and scope (state everything; Opus 4.8 will not infer these)

- **Prose, not bullet-slop.** Body sections are written in full academic paragraphs like AORE. Use tables only for the failure/mapping/results matrices and the error-model results. Do not render the paper as bulleted lists.
- **Intellectual honesty is the through-line.** Never claim literal 100% hallucination elimination as a result. The strong claim is a *conditional bound*; the headline finding is that correlated errors set a floor. Every empirical-sounding number must be labeled analytical/simulated or flagged as requiring live runs.
- **Do not invent citations, datasets, or results.** Mark gaps. Report a Monte-Carlo number only if it comes from the script you actually write and run; include the script in an appendix or state the repo placeholder.
- **Match AORE's demarcation discipline:** claim *integration/design* for the whole model and *empirical support only where an experiment backs it*, and state that the executed evidence is analytical simulation, not live agents.
- **Apply the math carefully.** Derive, don't assert. Show the binomial "≥k of N corroboration" survival expression, the correlation-adjusted version (e.g., beta-binomial or shared-latent-variable model), and the n-tier recursion, with assumptions named at each step. If a closed form isn't clean, give the recurrence and compute it numerically.
- **Verification step:** before finalizing, re-derive or numerically check every quantitative claim in the error model with a script, and confirm no invented citation slipped in.

## Output procedure

1. Do the related-work grounding research first (real citations).
2. Write and run the Monte-Carlo / analytical script for the error model; capture actual numbers and at least one figure.
3. Read the docx skill, then render the full paper to a .docx with proper headings, an abstract, numbered sections, tables, and the figure.
4. Deliver the .docx and a two-paragraph plain summary of what is proven analytically vs. what needs live runs.

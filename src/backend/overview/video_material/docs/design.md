# Design and motion direction

## A clear hierarchy of attention

The presentation uses an ink-blue stage inside a warm, restrained studio interface. Biological, psychological and social branches keep consistent blue, lavender and mint colours. Each person has a case accent, while the three ontology branch colours remain fixed.

Large Manrope headlines establish the idea of each chapter. Smaller labels explain what is visible; IBM Plex Mono identifies paths, phases, timings and technical quantities. Static font instances are included so browser and native-video typography agree. The original variable font, static instances and OFL licences remain in `assets/fonts`.

The scene engine works in a 1920 × 1080 coordinate system. Panels, connectors, typography and paths are drawn directly, so resolution changes retain crisp geometry. No raster illustration or stock photo is needed to explain the process.

## Sequential visual grammar

- **Opening:** a person and an expanding ontology constellation establish the human and structural scales.
- **Complaint:** the original text stays intact while exact spans are underlined. PERSON and CONTEXT receive their own panels.
- **Concept grounding:** six source-linked concepts appear in a readable grid. One exact CRITERION ancestry is traced at a time.
- **Hierarchy:** a staged reveal exposes real parent-child edges. A slow camera move reveals more structural detail.
- **Ancestry:** a tighter view isolates a featured option while its complete path appears as an indented sequence.
- **Observation model:** a bipartite graph makes many-to-many predictor-criterion hypotheses explicit. A draft, revision and pass sequence communicates the critic's role.
- **EMA:** check-in variables become progressively drawn time series.
- **Readiness:** a diagnostic gauge selects an analytical tier. Cases visibly diverge at this point.
- **Network and impact:** sufficient-data examples show associations; the sparse example remains a set of observations. Impact remains explicitly illustrative.
- **Breadth-first search:** successive phases illuminate different candidate paths, with particles travelling along actual parent-child edges.
- **Comparison:** the identical hierarchy is rendered twice, with a shared pool and budget. Counts derive from each selected set.
- **Targets:** ranked candidates are brought into readable cards alongside suitability and critic checks.
- **Fusion:** a two-part evidence bar makes the source weighting formula legible. Prior and updated illustrative ranks sit beside it.
- **HAPA:** a contrasting light delivery preview turns abstract targets into a brief action, surrounded by the barrier, phase, support and coping plan.
- **Memory:** a directed loop connects the approved model, observations, analysis, targets, intervention and history.
- **Closing:** the full proposition returns to the graph, with an explicit structural scope.

Transitions, particle positions, line reveals and camera moves are functions of absolute time. Scrubbing back to a moment yields the same frame, regardless of previous playback. Reduced motion removes camera travel, particles and scene entrance movement while keeping informative reveals.

## Legibility and hierarchy density

The rich default view is a deterministic, stratified sample across actual secondary domains, including the case candidate pool. It is a readable sample, not a claim to show every leaf at once. The full-catalogue control retains all 7,265 leaves. Ancestors are reconstructed from the same real paths in every view.

Most deep nodes are small and unlabelled at wide scale. Featured labels are shortened and positioned to avoid one another. The dedicated ancestry shot and interactive inspector reveal complete path text at a reading scale. The user can expose between 2 and 10 levels, choose a root branch and pan or zoom the hierarchy views.

## Comparison scope

A psychodynamic or CBT focus is a deliberately scoped example of a search policy. It represents neither an entire therapeutic tradition nor clinical practice in general. The integrative score-first option is included because conventional care can already combine domains, measurement and adaptation.

The quantities on the comparison panels are counts of secondary domains and authored criterion links. They are not percentages of clinical success. No outcome series, remission rate, causal estimate or claim of superiority is fabricated.

## Editing and reproduction

Change case copy and exact mapping queries in `data/cases.authoring.json`, rebuild the snapshot with `npm run data`, then run tests and render. Change shot timing in the `SHOTS` list in `app/model.js`; the browser timeline and renderer consume that same list. All scene drawing lives in `app/renderer.js`.

The all-scene tests exercise all five cases at multiple moments in every chapter. Poster files and 16-frame contact sheets provide a visual audit of the entire sequence. Short preview renders provide motion review without requiring a full re-export.

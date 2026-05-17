(() => {
  const appRoot = document.getElementById("session-app");
  if (!appRoot) return;

  const sessionId = appRoot.dataset.sessionId;
  const snapshotTag = document.getElementById("snapshot-json");
  const LOG_MAX_LINES = 1200;
  const VISIBILITY_STORAGE_PREFIX = "phoenix.visible";
  const VERBOSITY_STORAGE_KEY = "phoenix.ui.verbosity_mode";

  function parseSnapshot() {
    try {
      return JSON.parse(snapshotTag?.textContent || "{}");
    } catch {
      return {};
    }
  }

  const state = {
    snapshot: parseSnapshot(),
    activeSource: null,
    activeJobId: null,
    activeJobKind: "",
    activeTriggerButtonId: "",
    logLines: [],
    componentStatus: {},
    componentDetail: {},
    cycleRequestedRefinement: false,
    awaitingFreshAcquisition: false,
    impactChart: null,
    barrierChart: null,
    copingChart: null,
    timeSeriesChart: null,
    readinessComponentsChart: null,
    networkMethodChart: null,
    networkTemporalTrendChart: null,
    networkPredictorImportanceChart: null,
    impactDecompositionChart: null,
    stageRuntimeChart: null,
    stageStatusChart: null,
    timeSeriesData: null,
    timeSeriesCacheKey: "",
    logsDrawerOpen: false,
    controlDrawerOpen: false,
    sectionVisibilityUser: {},
    sectionVisibilityAuto: {},
    phaseState: [],
    engineFlowFromSummary: [],
    qualityFlowFromSummary: [],
    runtimeEvents: [],
    llmModelFetchTimer: null,
    llmModelLastQuery: "",
    llmModelLoaded: false,
    streamReconnectTimer: null,
    streamReconnectAttempts: 0,
    streamCursorByJob: {},
    lastTerminalJobMarker: "",
    snapshotRefreshTimer: null,
    snapshotRefreshPollTimer: null,
    snapshotRefreshInFlight: false,
    pendingSnapshotRefresh: false,
    lastSnapshotRefreshMs: 0,
    verbosityMode: "concise",
    networkAnimPayloadKey: "",
    networkAnimPayload: null,
    networkAnimPlaying: true,
    networkAnimFrameIndex: 0,
    networkAnimSpeed: 1,
    networkAnimLastTickMs: 0,
    networkAnimRafId: null,
    networkAnimPositions: {},
    reconcilingSnapshot: false,
  };

  const COMPONENT_DEFS = [
    { id: "full_session_pipeline", label: "Full Session Pipeline", category: "ORCHESTRATION" },
    { id: "cohort_batch", label: "Cohort Batch Runner", category: "ORCHESTRATION" },
    { id: "step01_operationalization", label: "Step 01 Operationalization", category: "MODEL CREATION" },
    { id: "step02_initial_model", label: "Step 02 Initial Model", category: "MODEL CREATION" },
    { id: "step02_visualization", label: "Step 02 Visual Diagnostics", category: "MODEL CREATION" },
    { id: "pseudodata_collection", label: "Data Collection / Pseudodata", category: "ACQUISITION" },
    { id: "manual_data_upload", label: "Manual Data Upload", category: "ACQUISITION" },
    { id: "pipeline_cycle_engine", label: "Cycle Pipeline", category: "ANALYSIS" },
    { id: "readiness_analysis", label: "Readiness Check", category: "ANALYSIS" },
    { id: "network_analysis", label: "Network Time-Series Analysis", category: "ANALYSIS" },
    { id: "impact_quantification", label: "Momentary Impact Quantification", category: "ANALYSIS" },
    { id: "step03_target_selection", label: "Step 03 Target Identification", category: "ANALYSIS" },
    { id: "step04_updated_model", label: "Step 04 Updated Model", category: "MODEL CREATION" },
    { id: "step05_intervention", label: "Step 05 Digital Intervention", category: "INTERVENTION" },
    { id: "impact_visualization", label: "Impact Visualization (Support)", category: "QUALITY + RESEARCH" },
    { id: "evaluation_reporting", label: "Research Reporting (Support)", category: "QUALITY + RESEARCH" },
    { id: "communication_agent", label: "Communication Agent", category: "INTERVENTION" },
  ];

  const JOB_KIND_COMPONENTS = {
    full_session_pipeline: [
      "full_session_pipeline",
      "step01_operationalization",
      "step02_initial_model",
      "step02_visualization",
      "pseudodata_collection",
      "pipeline_cycle_engine",
      "readiness_analysis",
      "network_analysis",
      "impact_quantification",
      "step03_target_selection",
      "step04_updated_model",
      "step05_intervention",
      "impact_visualization",
      "evaluation_reporting",
    ],
    initial_model: ["step01_operationalization", "step02_initial_model", "step02_visualization"],
    synthesize_pseudodata: ["pseudodata_collection"],
    manual_data_upload: ["manual_data_upload"],
    pipeline_cycle: [
      "pipeline_cycle_engine",
      "readiness_analysis",
      "network_analysis",
      "impact_quantification",
      "step03_target_selection",
      "step04_updated_model",
      "step05_intervention",
      "impact_visualization",
      "evaluation_reporting",
    ],
    full_cohort: [
      "cohort_batch",
      "step01_operationalization",
      "step02_initial_model",
      "step02_visualization",
      "pseudodata_collection",
      "pipeline_cycle_engine",
      "readiness_analysis",
      "network_analysis",
      "impact_quantification",
      "step03_target_selection",
      "step04_updated_model",
      "step05_intervention",
      "impact_visualization",
      "evaluation_reporting",
    ],
  };

  const UI = {
    runtimeGrid: document.getElementById("runtime-component-grid"),
    logConsole: document.getElementById("log-console"),
    activeJobId: document.getElementById("active-job-id"),
    activeJobStatus: document.getElementById("active-job-status"),
    activeComponent: document.getElementById("active-component"),
    drawerActiveJobId: document.getElementById("drawer-active-job-id"),
    drawerActiveJobStatus: document.getElementById("drawer-active-job-status"),
    errorBanner: document.getElementById("job-error-banner"),
    logsDrawer: document.getElementById("logs-drawer"),
    logsDrawerOpenBtn: document.getElementById("logs-drawer-open"),
    logsDrawerCloseBtn: document.getElementById("logs-drawer-close"),
    controlDrawer: document.getElementById("control-drawer"),
    controlDrawerOpenBtn: document.getElementById("control-drawer-open"),
    controlDrawerCloseBtn: document.getElementById("control-drawer-close"),
    openLogsFromControlBtn: document.getElementById("open-logs-from-control"),
    drawerBackdrop: document.getElementById("drawer-backdrop"),
    sectionToggles: Array.from(document.querySelectorAll(".section-visibility-toggle")),
    runNextPhaseBtn: document.getElementById("run-next-phase-btn"),
    topbarPipelineStrip: document.getElementById("topbar-pipeline-strip"),
    topbarPipelineNodes: document.getElementById("topbar-pipeline-nodes"),
    runtimeStageSummary: document.getElementById("runtime-stage-summary"),
    runtimeLastEvent: document.getElementById("runtime-last-event"),
    runtimeEventsList: document.getElementById("runtime-events-list"),
    flowInputChannel: document.getElementById("flow-input-channel"),
    flowInputDetail: document.getElementById("flow-input-detail"),
    flowExecutionChannel: document.getElementById("flow-execution-channel"),
    flowExecutionDetail: document.getElementById("flow-execution-detail"),
    flowStageGrid: document.getElementById("flow-stage-grid"),
    flowNextChip: document.getElementById("flow-next-chip"),
    flowNextAction: document.getElementById("flow-next-action"),
    flowRunNextBtn: document.getElementById("flow-run-next-btn"),
    flowOpenLogsBtn: document.getElementById("flow-open-logs-btn"),
    flowProgressFill: document.getElementById("flow-progress-fill"),
    flowProgressLabel: document.getElementById("flow-progress-label"),
    flowProgressDetail: document.getElementById("flow-progress-detail"),
    flowActiveComponent: document.getElementById("flow-active-component"),
    flowCycleLoop: document.getElementById("flow-cycle-loop"),
    flowIntermediateFeed: document.getElementById("flow-intermediate-feed"),
    verbosityModeSelect: document.getElementById("ui-verbosity-mode"),
    flowVerbosityHint: document.getElementById("flow-verbosity-hint"),
    networkAnimCanvas: document.getElementById("network-anim-canvas"),
    networkAnimStatus: document.getElementById("network-anim-status"),
    networkAnimToggle: document.getElementById("network-anim-toggle"),
    networkAnimFrame: document.getElementById("network-anim-frame"),
    networkAnimSpeed: document.getElementById("network-anim-speed"),
    networkAnimMeta: document.getElementById("network-anim-meta"),
    networkAnimCaption: document.getElementById("network-anim-caption"),
    networkAnimEdges: document.getElementById("network-anim-edges"),
    cohortSummary: document.getElementById("cohort-summary"),
    llmModelOptions: document.getElementById("llm-model-options"),
    llmModelInputs: [
      document.getElementById("full-llm-model"),
      document.getElementById("initial-llm-model"),
      document.getElementById("cycle-llm-model"),
      document.getElementById("cohort-llm-model"),
    ].filter(Boolean),
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function shortPath(value, keepSegments = 3) {
    const raw = String(value || "").trim();
    if (!raw) return "—";
    const norm = raw.replaceAll("\\", "/");
    const parts = norm.split("/").filter((item) => item.length > 0);
    if (parts.length <= keepSegments) return norm;
    return `.../${parts.slice(-keepSegments).join("/")}`;
  }

  function normalizeStatus(status) {
    const raw = String(status || "idle").toLowerCase();
    if (["idle", "queued", "running", "succeeded", "failed"].includes(raw)) return raw;
    if (raw === "done") return "succeeded";
    if (raw === "error") return "failed";
    return "idle";
  }

  function normalizePipelineSummaryStatus(status) {
    const raw = String(status || "").trim().toLowerCase();
    if (!raw) return "idle";
    if (["succeeded", "success", "done", "completed", "complete", "ok"].includes(raw)) return "succeeded";
    if (["failed", "failure", "error", "crashed"].includes(raw)) return "failed";
    if (["running", "in_progress", "in-progress", "queued", "started"].includes(raw)) return "running";
    return normalizeStatus(raw);
  }

  function applyStatusClass(node, status) {
    if (!node) return;
    const classes = ["status-idle", "status-queued", "status-running", "status-succeeded", "status-failed"];
    classes.forEach((cls) => node.classList.remove(cls));
    node.classList.add(`status-${normalizeStatus(status)}`);
  }

  function isRunningStatus(status) {
    return ["queued", "running"].includes(normalizeStatus(status));
  }

  function normalizeVerbosityMode(mode) {
    const token = String(mode || "").trim().toLowerCase();
    if (["concise", "balanced", "detailed"].includes(token)) return token;
    return "concise";
  }

  function shouldSuppressLogLine(line) {
    const mode = normalizeVerbosityMode(state.verbosityMode);
    if (mode === "detailed") return false;
    const low = String(line || "").toLowerCase();
    if (!low) return false;
    if (low.includes("[frontend]")) return false;
    if (low.includes("failed") || low.includes("error") || low.includes("exception") || low.includes("traceback")) {
      return false;
    }
    if (low.includes("heartbeat elapsed=")) return true;
    if (mode === "balanced") return false;
    if (
      low.includes("llm call start")
      || low.includes("llm call ok")
      || low.includes("auto-repair: attempt")
      || low.includes("critic loop")
      || low.includes("command_start cwd=")
      || low.includes("still running")
    ) {
      return true;
    }
    return false;
  }

  function setVerbosityMode(mode, persist = true) {
    const next = normalizeVerbosityMode(mode);
    state.verbosityMode = next;
    document.body.setAttribute("data-verbosity-mode", next);
    if (UI.verbosityModeSelect && UI.verbosityModeSelect.value !== next) {
      UI.verbosityModeSelect.value = next;
    }
    if (UI.flowVerbosityHint) {
      if (next === "concise") {
        UI.flowVerbosityHint.textContent = "Concise hides heartbeat and low-signal intermediate noise.";
      } else if (next === "balanced") {
        UI.flowVerbosityHint.textContent = "Balanced shows stage transitions with reduced heartbeat noise.";
      } else {
        UI.flowVerbosityHint.textContent = "Detailed shows full runtime stream, including intermediate traces.";
      }
    }
    if (persist) {
      try {
        localStorage.setItem(VERBOSITY_STORAGE_KEY, next);
      } catch {
        // Ignore storage failures; UI still updates in-memory.
      }
    }
  }

  function appendLog(line) {
    if (!UI.logConsole) return;
    const text = String(line || "");
    if (shouldSuppressLogLine(text)) return;
    state.logLines.push(text);
    while (state.logLines.length > LOG_MAX_LINES) state.logLines.shift();
    UI.logConsole.textContent = state.logLines.join("\n");
    UI.logConsole.scrollTop = UI.logConsole.scrollHeight;
  }

  function renderRuntimeEvents() {
    if (!UI.runtimeEventsList) return;
    const maxRows = state.verbosityMode === "detailed" ? 8 : state.verbosityMode === "balanced" ? 6 : 4;
    const rows = state.runtimeEvents.slice(0, maxRows);
    if (!rows.length) {
      UI.runtimeEventsList.innerHTML = "";
      return;
    }
    UI.runtimeEventsList.innerHTML = rows.map((item) => `
      <li><code>${escapeHtml(item.time || "--:--:--")}</code> ${escapeHtml(item.message || "")}</li>
    `).join("");
  }

  function pushRuntimeEvent(message) {
    const raw = String(message || "").trim();
    if (!raw) return;
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, "0");
    const mm = String(now.getMinutes()).padStart(2, "0");
    const ss = String(now.getSeconds()).padStart(2, "0");
    state.runtimeEvents.unshift({ time: `${hh}:${mm}:${ss}`, message: raw.slice(0, 220) });
    while (state.runtimeEvents.length > 20) state.runtimeEvents.pop();
    if (UI.runtimeLastEvent) UI.runtimeLastEvent.textContent = raw.slice(0, 220);
    renderRuntimeEvents();
  }

  function setLiveRefreshStatus(text, status = "idle") {
    const node = document.getElementById("dashboard-live-refresh-status");
    if (!node) return;
    const safe = normalizeStatus(status);
    node.textContent = String(text || "");
    node.className = `status-chip status-${safe}`;
  }

  function clearLiveSnapshotRefreshTimer() {
    if (state.snapshotRefreshTimer) {
      clearTimeout(state.snapshotRefreshTimer);
      state.snapshotRefreshTimer = null;
    }
  }

  function clearLiveSnapshotPollTimer() {
    if (state.snapshotRefreshPollTimer) {
      clearInterval(state.snapshotRefreshPollTimer);
      state.snapshotRefreshPollTimer = null;
    }
  }

  function scheduleSnapshotRefresh(reason = "", force = false) {
    const minIntervalMs = 2600;
    if (state.snapshotRefreshInFlight) {
      state.pendingSnapshotRefresh = true;
      return;
    }
    if (state.snapshotRefreshTimer) return;
    const now = Date.now();
    const elapsed = now - Number(state.lastSnapshotRefreshMs || 0);
    const delayMs = force ? 0 : Math.max(0, minIntervalMs - elapsed);
    state.snapshotRefreshTimer = window.setTimeout(async () => {
      state.snapshotRefreshTimer = null;
      if (state.snapshotRefreshInFlight) {
        state.pendingSnapshotRefresh = true;
        return;
      }
      state.snapshotRefreshInFlight = true;
      setLiveRefreshStatus("LIVE SYNC…", "running");
      try {
        await refreshSnapshot();
        state.lastSnapshotRefreshMs = Date.now();
        const syncLabel = new Date(state.lastSnapshotRefreshMs).toLocaleTimeString();
        const suffix = reason ? ` (${reason})` : "";
        setLiveRefreshStatus(`SYNCED ${syncLabel}${suffix}`, "succeeded");
      } catch (err) {
        appendLog(`[frontend] Live snapshot refresh failed: ${err.message || err}`);
        setLiveRefreshStatus("SYNC ERROR", "failed");
      } finally {
        state.snapshotRefreshInFlight = false;
        if (state.pendingSnapshotRefresh) {
          state.pendingSnapshotRefresh = false;
          scheduleSnapshotRefresh("queued", false);
        }
      }
    }, delayMs);
  }

  function startLiveSnapshotPolling() {
    clearLiveSnapshotPollTimer();
    state.snapshotRefreshPollTimer = window.setInterval(() => {
      if (!state.activeJobId) return;
      scheduleSnapshotRefresh("poll", false);
    }, 10000);
  }

  function updateBackdrop() {
    if (!UI.drawerBackdrop) return;
    const open = state.logsDrawerOpen || state.controlDrawerOpen;
    UI.drawerBackdrop.hidden = !open;
    UI.drawerBackdrop.classList.toggle("is-open", open);
  }

  function setLogsDrawerOpen(open) {
    state.logsDrawerOpen = Boolean(open);
    if (!UI.logsDrawer) return;
    UI.logsDrawer.classList.toggle("is-open", state.logsDrawerOpen);
    UI.logsDrawer.setAttribute("aria-hidden", state.logsDrawerOpen ? "false" : "true");
    updateBackdrop();
  }

  function setControlDrawerOpen(open) {
    state.controlDrawerOpen = Boolean(open);
    if (!UI.controlDrawer) return;
    UI.controlDrawer.classList.toggle("is-open", state.controlDrawerOpen);
    UI.controlDrawer.setAttribute("aria-hidden", state.controlDrawerOpen ? "false" : "true");
    updateBackdrop();
  }

  function closeDrawers() {
    setLogsDrawerOpen(false);
    setControlDrawerOpen(false);
  }

  function setJobMeta(jobId, status) {
    const safe = normalizeStatus(status);
    if (UI.activeJobId) UI.activeJobId.textContent = jobId || "none";
    if (UI.activeJobStatus) {
      UI.activeJobStatus.textContent = safe.toUpperCase();
      applyStatusClass(UI.activeJobStatus, safe);
    }
    if (UI.drawerActiveJobId) UI.drawerActiveJobId.textContent = jobId || "none";
    if (UI.drawerActiveJobStatus) {
      UI.drawerActiveJobStatus.textContent = safe.toUpperCase();
      applyStatusClass(UI.drawerActiveJobStatus, safe);
    }
  }

  function setActiveComponent(componentId, status = "running") {
    const safeStatus = normalizeStatus(status);
    const label = COMPONENT_DEFS.find((item) => item.id === componentId)?.label || componentId || "idle";
    if (UI.activeComponent) {
      UI.activeComponent.textContent = safeStatus === "idle" ? "idle" : label;
      applyStatusClass(UI.activeComponent, safeStatus);
    }
    if (UI.runtimeStageSummary) {
      const suffix = safeStatus === "idle" ? "" : ` (${safeStatus.toUpperCase()})`;
      UI.runtimeStageSummary.textContent = safeStatus === "idle" ? "Idle" : `${label}${suffix}`;
    }
  }

  function setComponentState(componentId, status, detail = "") {
    const safe = normalizeStatus(status);
    const previousStatus = state.componentStatus[componentId];
    const previousDetail = state.componentDetail[componentId];
    state.componentStatus[componentId] = safe;
    if (detail) state.componentDetail[componentId] = detail;
    const card = UI.runtimeGrid?.querySelector(`[data-component="${componentId}"]`);
    if (card) {
      card.dataset.status = safe;
      const chip = card.querySelector(".status-chip");
      const detailNode = card.querySelector(".runtime-card-detail");
      if (chip) {
        chip.textContent = safe.toUpperCase();
        applyStatusClass(chip, safe);
      }
      if (detailNode) detailNode.textContent = detail || state.componentDetail[componentId] || "—";
    }
    if (safe !== previousStatus || (detail && detail !== previousDetail)) {
      const label = COMPONENT_DEFS.find((item) => item.id === componentId)?.label || componentId;
      const suffix = detail ? `: ${detail}` : "";
      pushRuntimeEvent(`${label} -> ${safe.toUpperCase()}${suffix}`);
    }
    syncRuntimeCardVisibility();
    if (!state.reconcilingSnapshot) {
      renderPhaseProgress();
    }
  }

  function shouldShowRuntimeCard(componentId) {
    if (componentId !== "communication_agent") return true;
    const status = normalizeStatus(state.componentStatus[componentId] || "idle");
    if (status !== "idle") return true;
    const stage = String(state.snapshot?.communication_summary?.payload?.stage || "").toLowerCase();
    return stage.startsWith("cycle_");
  }

  function syncRuntimeCardVisibility() {
    if (!UI.runtimeGrid) return;
    COMPONENT_DEFS.forEach((item) => {
      const card = UI.runtimeGrid.querySelector(`[data-component="${item.id}"]`);
      if (!card) return;
      card.classList.toggle("runtime-card-hidden", !shouldShowRuntimeCard(item.id));
    });
  }

  function initRuntimeMap() {
    if (!UI.runtimeGrid) return;
    UI.runtimeGrid.innerHTML = COMPONENT_DEFS.map((item) => `
      <article class="runtime-card" data-component="${item.id}" data-status="idle">
        <div class="runtime-card-head">
          <span class="runtime-card-title">${escapeHtml(item.label)}</span>
          <span class="badge status-chip status-idle">IDLE</span>
        </div>
        <div class="runtime-card-detail">Category: ${escapeHtml(item.category)}</div>
      </article>
    `).join("");
    COMPONENT_DEFS.forEach((item) => {
      state.componentStatus[item.id] = "idle";
      state.componentDetail[item.id] = `Category: ${item.category}`;
    });
    syncRuntimeCardVisibility();
  }

  function sectionVisibilityStorageKey(key) {
    return `${VISIBILITY_STORAGE_PREFIX}.${sessionId}.${key}`;
  }

  function getUserVisibilitySetting(key) {
    if (Object.prototype.hasOwnProperty.call(state.sectionVisibilityUser, key)) {
      return state.sectionVisibilityUser[key];
    }
    const stored = localStorage.getItem(sectionVisibilityStorageKey(key));
    if (stored === "hidden") return false;
    if (stored === "visible") return true;
    return true;
  }

  function setUserVisibilitySetting(key, visible) {
    state.sectionVisibilityUser[key] = Boolean(visible);
    localStorage.setItem(sectionVisibilityStorageKey(key), visible ? "visible" : "hidden");
  }

  function setSectionVisible(key, visible) {
    const section = document.querySelector(`.collapsible-section[data-section-key="${key}"]`);
    if (!section) return;
    section.classList.toggle("hidden", !visible);
  }

  function applySectionVisibility() {
    const togglesByKey = new Map();
    UI.sectionToggles.forEach((toggle) => togglesByKey.set(toggle.dataset.sectionTarget || "", toggle));

    const allKeys = new Set([
      ...Array.from(togglesByKey.keys()),
      ...Array.from(document.querySelectorAll(".collapsible-section")).map((node) => node.dataset.sectionKey || ""),
    ]);

    allKeys.forEach((key) => {
      if (!key) return;
      const autoVisible = state.sectionVisibilityAuto[key] !== false;
      const userVisible = getUserVisibilitySetting(key);
      const finalVisible = autoVisible && userVisible;
      setSectionVisible(key, finalVisible);
      const toggle = togglesByKey.get(key);
      if (toggle) {
        toggle.checked = userVisible;
        toggle.disabled = !autoVisible;
        toggle.title = autoVisible ? "" : "This panel will appear when prerequisite outputs are available.";
      }
    });
  }

  function setCollapsed(section, collapsed, persist = true) {
    const key = section.dataset.sectionKey || "";
    const body = section.querySelector(".section-body");
    const btn = section.querySelector(".toggle-section-btn");
    if (!body || !btn) return;
    body.classList.toggle("is-collapsed", collapsed);
    btn.textContent = collapsed ? "Expand" : "Collapse";
    if (persist && key) {
      localStorage.setItem(`phoenix.section.${sessionId}.${key}`, collapsed ? "collapsed" : "open");
    }
  }

  function initCollapsibleSections() {
    const sections = Array.from(document.querySelectorAll(".collapsible-section"));
    sections.forEach((section) => {
      const key = section.dataset.sectionKey || "";
      const defaultOpen = String(section.dataset.defaultOpen || "true").toLowerCase() === "true";
      const stored = key ? localStorage.getItem(`phoenix.section.${sessionId}.${key}`) : null;
      const collapsed = stored ? stored === "collapsed" : !defaultOpen;
      setCollapsed(section, collapsed, false);
      section.querySelector(".toggle-section-btn")?.addEventListener("click", () => {
        const isCollapsed = section.querySelector(".section-body")?.classList.contains("is-collapsed");
        setCollapsed(section, !isCollapsed);
      });
    });

    document.getElementById("expand-all-btn")?.addEventListener("click", () => {
      sections.forEach((section) => setCollapsed(section, false));
    });
    document.getElementById("collapse-all-btn")?.addEventListener("click", () => {
      sections.forEach((section) => setCollapsed(section, true));
    });
  }

  function initSectionVisibilityControls() {
    UI.sectionToggles.forEach((toggle) => {
      const key = toggle.dataset.sectionTarget || "";
      if (!key) return;
      toggle.checked = getUserVisibilitySetting(key);
      toggle.addEventListener("change", () => {
        setUserVisibilitySetting(key, Boolean(toggle.checked));
        applySectionVisibility();
      });
    });
  }

  function showError(message) {
    if (!UI.errorBanner) return;
    const raw = String(message || "").trim();
    if (!raw) {
      UI.errorBanner.classList.add("hidden");
      UI.errorBanner.textContent = "";
      return;
    }
    const low = raw.toLowerCase();
    let hint = "Inspect realtime logs for the failing component and rerun.";
    if (low.includes("collect_allowed_predictor_paths")) {
      hint = "Step 02 guardrail argument mismatch detected. Pull latest code and rerun model creation.";
      setComponentState("step02_initial_model", "failed", "Guardrail argument mismatch");
    } else if (low.includes("no module named 'shared'")) {
      hint = "Step 02 import bootstrap failure. Pull latest code and rerun model creation.";
      setComponentState("step02_initial_model", "failed", "Import bootstrap failure");
    } else if (low.includes("produced no model artifact") || low.includes("model json is missing")) {
      hint = "Step 02 worker failed before output serialization. Check worker error lines in logs.";
      setComponentState("step02_initial_model", "failed", "Model artifact missing");
    } else if (low.includes("no module named 'pandas'")) {
      hint = "Python dependency missing. Install project requirements in the selected environment.";
    } else if (low.includes("cohort run finished with failures")) {
      hint = "Some cohort patients failed. Open logs and inspect the cohort manifest path in the error message.";
      setComponentState("cohort_batch", "failed", "One or more patients failed");
    } else if (low.includes("session already has running job")) {
      hint = "A job is already running for this session. Wait for completion before launching a new one.";
    }
    UI.errorBanner.innerHTML = `<strong>Runtime error:</strong> ${escapeHtml(raw)}<br><span>${escapeHtml(hint)}</span>`;
    UI.errorBanner.classList.remove("hidden");
    if (typeof switchTab === 'function') switchTab('logs');
  }

  function setButtonLoading(buttonId, loading, runningLabel = "Processing…") {
    const btn = buttonId ? document.getElementById(buttonId) : null;
    if (!btn) return;
    if (!btn.dataset.defaultLabel) btn.dataset.defaultLabel = btn.textContent || "";
    btn.textContent = loading ? runningLabel : btn.dataset.defaultLabel;
    if (loading) btn.disabled = true;
  }

  function syncControlStates() {
    const busy = Boolean(state.activeSource);
    const hasModel = Boolean(state.snapshot.has_model);
    const hasPseudodata = Boolean(state.snapshot.has_pseudodata) && !state.awaitingFreshAcquisition;
    const setDisabled = (id, disabled, title = "") => {
      const node = document.getElementById(id);
      if (!node) return;
      if (state.activeTriggerButtonId && id === state.activeTriggerButtonId && busy) return;
      node.disabled = Boolean(disabled);
      node.title = title;
    };
    setDisabled("run-full-btn", busy, busy ? "A background job is running." : "");
    setDisabled("save-intake-btn", busy, busy ? "A background job is running." : "");
    setDisabled("run-initial-model-btn", busy, busy ? "A background job is running." : "");
    setDisabled("synthesize-btn", busy || !hasModel, !hasModel ? "Run model creation first." : "");
    setDisabled("manual-upload-btn", busy || !hasModel, !hasModel ? "Run model creation first." : "");
    setDisabled(
      "run-cycle-btn",
      busy || !hasModel || !hasPseudodata,
      !hasPseudodata ? "Acquire pseudodata first." : "",
    );
    setDisabled("flow-run-next-btn", busy, busy ? "A background job is running." : "");
    setDisabled("run-cohort-btn", busy, busy ? "A background job is running." : "");
  }

  async function apiGet(path) {
    const res = await fetch(path);
    const payload = await res.json();
    if (!res.ok || payload.status === "error") {
      throw new Error(payload.message || `HTTP ${res.status}`);
    }
    return payload;
  }

  async function apiPost(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const payload = await res.json();
    if (!res.ok || payload.status === "error") {
      throw new Error(payload.message || `HTTP ${res.status}`);
    }
    return payload;
  }

  function renderLlmModelOptions(models) {
    const dropdown = document.getElementById("llm-autocomplete-dropdown");
    if (!dropdown) return;
    const rows = Array.isArray(models) ? models : [];
    state._llmCatalog = rows;
    if (!rows.length) {
      dropdown.innerHTML = `<div class="autocomplete-item muted">No models found</div>`;
      return;
    }
    dropdown.innerHTML = rows.slice(0, 40).map((row) => {
      const id = String(row.id || "").trim();
      if (!id) return "";
      return `<div class="autocomplete-item" data-model-id="${escapeHtml(id)}"><span class="model-id">${escapeHtml(id)}</span></div>`;
    }).join("");
  }

  function showLlmDropdown() {
    const dropdown = document.getElementById("llm-autocomplete-dropdown");
    if (dropdown) dropdown.classList.remove("hidden");
  }
  function hideLlmDropdown() {
    setTimeout(() => {
      const dropdown = document.getElementById("llm-autocomplete-dropdown");
      if (dropdown) dropdown.classList.add("hidden");
    }, 180);
  }

  async function fetchLlmModelCatalog(query = "") {
    const q = String(query || "").trim();
    const response = await apiGet(`/api/llm/models?q=${encodeURIComponent(q)}&limit=120`);
    const rows = Array.isArray(response.models) ? response.models : [];
    if (rows.length) {
      renderLlmModelOptions(rows);
      state.llmModelLoaded = true;
      state.llmModelLastQuery = q;
    }
  }

  function scheduleLlmModelFetch(query = "") {
    if (state.llmModelFetchTimer) {
      clearTimeout(state.llmModelFetchTimer);
      state.llmModelFetchTimer = null;
    }
    const q = String(query || "").trim();
    state.llmModelFetchTimer = window.setTimeout(() => {
      fetchLlmModelCatalog(q).catch((err) => {
        appendLog(`[frontend] model catalog lookup failed: ${err.message || err}`);
      });
    }, 140);
  }

  function bindModelCatalogAutocomplete() {
    const inputs = UI.llmModelInputs || [];
    if (!inputs.length) return;

    /* Click handler on the dropdown */
    const dropdown = document.getElementById("llm-autocomplete-dropdown");
    if (dropdown) {
      dropdown.addEventListener("mousedown", (e) => {
        e.preventDefault(); /* prevent blur */
        const item = e.target.closest(".autocomplete-item");
        if (!item) return;
        const modelId = item.dataset.modelId;
        if (!modelId) return;
        /* Set ALL visible LLM model inputs to the selected model */
        inputs.forEach((inp) => { if (inp) inp.value = modelId; });
        hideLlmDropdown();
      });
    }

    inputs.forEach((node) => {
      node?.addEventListener("focus", () => {
        if (!state.llmModelLoaded) scheduleLlmModelFetch("");
        showLlmDropdown();
      });
      node?.addEventListener("blur", hideLlmDropdown);
      node?.addEventListener("input", (event) => {
        const value = String(event?.target?.value || "");
        if (value.length >= 1) {
          scheduleLlmModelFetch(value);
        } else if (!state.llmModelLoaded) {
          scheduleLlmModelFetch("");
        }
        showLlmDropdown();
      });
    });
    scheduleLlmModelFetch("");
  }

  function readBaselines() {
    const rows = [];
    document.querySelectorAll(".baseline-input").forEach((node) => {
      rows.push({ var_id: node.dataset.varId, baseline_0_1: Number(node.value) });
    });
    return rows;
  }

  function destroyCharts() {
    [
      state.impactChart,
      state.barrierChart,
      state.copingChart,
      state.timeSeriesChart,
      state.readinessComponentsChart,
      state.networkMethodChart,
      state.networkTemporalTrendChart,
      state.networkPredictorImportanceChart,
      state.impactDecompositionChart,
      state.stageRuntimeChart,
      state.stageStatusChart,
    ].forEach((chart) => {
      if (chart && typeof chart.destroy === "function") chart.destroy();
    });
    state.impactChart = null;
    state.barrierChart = null;
    state.copingChart = null;
    state.timeSeriesChart = null;
    state.readinessComponentsChart = null;
    state.networkMethodChart = null;
    state.networkTemporalTrendChart = null;
    state.networkPredictorImportanceChart = null;
    state.impactDecompositionChart = null;
    state.stageRuntimeChart = null;
    state.stageStatusChart = null;
  }

  function stopNetworkAnimationLoop() {
    if (state.networkAnimRafId) {
      cancelAnimationFrame(state.networkAnimRafId);
      state.networkAnimRafId = null;
    }
    state.networkAnimLastTickMs = 0;
  }

  function setNetworkAnimStatus(text, status = "idle") {
    if (!UI.networkAnimStatus) return;
    UI.networkAnimStatus.textContent = String(text || "");
    UI.networkAnimStatus.className = `status-chip status-${normalizeStatus(status)}`;
  }

  function networkPayloadIdentity(payload) {
    const summary = payload?.summary || {};
    const frameCount = Number(summary.frame_count || (payload?.frames || []).length || 0);
    const nodeCount = Number(summary.node_count || (payload?.nodes || []).length || 0);
    const edgeCount = Number(summary.edge_count || (payload?.edges || []).length || 0);
    const first = (payload?.frames || [])[0] || {};
    const last = (payload?.frames || [])[Math.max(0, (payload?.frames || []).length - 1)] || {};
    return [frameCount, nodeCount, edgeCount, first.time_index ?? 0, last.time_index ?? 0].join("|");
  }

  function roleColor(role) {
    const token = String(role || "").toLowerCase();
    if (token === "criterion") return "#f9a66b";
    if (token === "predictor") return "#67d7f7";
    return "#b8cce9";
  }

  function directionColor(direction) {
    const token = String(direction || "").toLowerCase();
    if (token === "decrease" || token === "down") return [244, 180, 59];
    return [90, 220, 255];
  }

  function isDashboardTabActive() {
    const pane = document.getElementById("tab-dashboard");
    return Boolean(pane && pane.classList.contains("active"));
  }

  function updateNetworkAnimationStatus(payload) {
    const frames = Array.isArray(payload?.frames) ? payload.frames : [];
    if (!payload || !frames.length) {
      setNetworkAnimStatus("UNAVAILABLE", "idle");
      return;
    }
    if (frames.length === 1) {
      setNetworkAnimStatus("STATIC", "succeeded");
      return;
    }
    setNetworkAnimStatus(state.networkAnimPlaying ? "ANIMATING" : "PAUSED", state.networkAnimPlaying ? "running" : "queued");
  }

  function ensureNetworkCanvasSize() {
    const canvas = UI.networkAnimCanvas;
    if (!canvas) return { canvas: null, width: 0, height: 0, ctx: null };
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const width = Math.max(300, Math.floor(rect.width));
    const height = Math.max(220, Math.floor(rect.height));
    if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return { canvas, width, height, ctx: null };
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { canvas, width, height, ctx };
  }

  function computeNetworkNodePositions(nodes, width, height) {
    const criteria = nodes.filter((row) => String(row.role || "").toLowerCase() === "criterion");
    const predictors = nodes.filter((row) => String(row.role || "").toLowerCase() === "predictor");
    const unknown = nodes.filter((row) => {
      const token = String(row.role || "").toLowerCase();
      return token !== "criterion" && token !== "predictor";
    });
    const out = {};
    const layoutColumn = (rows, x, startY, endY, wobble = 0) => {
      if (!rows.length) return;
      const denom = Math.max(1, rows.length - 1);
      rows.forEach((row, idx) => {
        const y = rows.length === 1
          ? (startY + endY) / 2
          : (startY + ((endY - startY) * (idx / denom)));
        out[String(row.id)] = {
          x: x + Math.sin((idx + 1) * 0.9) * wobble,
          y,
        };
      });
    };
    layoutColumn(criteria, width * 0.24, height * 0.16, height * 0.86, 10);
    layoutColumn(predictors, width * 0.76, height * 0.16, height * 0.86, 10);
    if (unknown.length) {
      const cx = width * 0.5;
      const cy = height * 0.5;
      const ring = Math.max(50, Math.min(width, height) * 0.22);
      unknown.forEach((row, idx) => {
        const angle = (Math.PI * 2 * idx) / Math.max(1, unknown.length);
        out[String(row.id)] = {
          x: cx + Math.cos(angle) * ring,
          y: cy + Math.sin(angle) * ring,
        };
      });
    }
    return out;
  }

  function normalizeNetworkPayload(raw) {
    const payload = raw && typeof raw === "object" ? raw : {};
    const nodes = Array.isArray(payload.nodes)
      ? payload.nodes
        .map((row) => ({
          id: String(row?.id || "").trim(),
          label: String(row?.label || row?.id || "").trim(),
          role: String(row?.role || "unknown").trim().toLowerCase(),
        }))
        .filter((row) => row.id)
      : [];
    const frames = Array.isArray(payload.frames)
      ? payload.frames
        .map((row) => ({
          time_index: Number(row?.time_index || 0),
          t: Number(row?.t || 0),
          node_strength_abs: row?.node_strength_abs && typeof row.node_strength_abs === "object" ? row.node_strength_abs : {},
          node_betweenness: row?.node_betweenness && typeof row.node_betweenness === "object" ? row.node_betweenness : {},
        }))
        .sort((a, b) => Number(a.time_index || 0) - Number(b.time_index || 0))
      : [];
    const edges = Array.isArray(payload.edges)
      ? payload.edges
        .map((row) => ({
          src: String(row?.src || "").trim(),
          dst: String(row?.dst || "").trim(),
          abs_change: Number(row?.abs_change || 0),
          norm_change: Number(row?.norm_change || 0),
          peak_time_index: Number(row?.peak_time_index || 0),
          peak_t: Number(row?.peak_t || 0),
          direction: String(row?.direction || "").trim().toLowerCase(),
          range: Number(row?.range || 0),
          slope: Number(row?.slope || 0),
        }))
        .filter((row) => row.src && row.dst && row.src !== row.dst)
      : [];
    const maxStrength = (() => {
      let maxValue = 0;
      frames.forEach((frame) => {
        const entries = frame.node_strength_abs || {};
        Object.values(entries).forEach((value) => {
          const score = Number(value || 0);
          if (score > maxValue) maxValue = score;
        });
      });
      return maxValue > 0 ? maxValue : 1;
    })();
    return {
      status: String(payload.status || "").trim().toLowerCase(),
      reason: String(payload.reason || "").trim(),
      summary: payload.summary && typeof payload.summary === "object" ? payload.summary : {},
      nodes,
      frames,
      edges,
      maxStrength,
    };
  }

  function drawNetworkAnimationFrame(payload, frameIndex) {
    const prepared = ensureNetworkCanvasSize();
    const ctx = prepared.ctx;
    const width = prepared.width;
    const height = prepared.height;
    if (!ctx || !width || !height) return;

    const frames = payload.frames || [];
    if (!frames.length || !(payload.nodes || []).length) {
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "rgba(185, 203, 232, 0.85)";
      ctx.font = "600 14px Space Grotesk, sans-serif";
      ctx.fillText("No time-varying network frames available.", 18, 30);
      return;
    }

    const safeIndex = Math.max(0, Math.min(frames.length - 1, Number(frameIndex || 0)));
    const frame = frames[safeIndex] || {};
    const nodes = payload.nodes || [];
    const edges = payload.edges || [];
    const nodeStrength = frame.node_strength_abs || {};
    const nodeBetweenness = frame.node_betweenness || {};
    const timeIndex = Number(frame.time_index || safeIndex);
    const timeNorm = Number(frame.t || 0);

    if (!state.networkAnimPositions || Object.keys(state.networkAnimPositions).length !== nodes.length) {
      state.networkAnimPositions = computeNetworkNodePositions(nodes, width, height);
    }
    const positions = state.networkAnimPositions || {};

    const bgGrad = ctx.createLinearGradient(0, 0, width, height);
    bgGrad.addColorStop(0, "rgba(12, 22, 38, 0.94)");
    bgGrad.addColorStop(1, "rgba(8, 14, 26, 0.98)");
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height);

    const activeEdgeRows = [];
    edges.forEach((edge) => {
      const srcPos = positions[String(edge.src || "")];
      const dstPos = positions[String(edge.dst || "")];
      if (!srcPos || !dstPos) return;
      const norm = Math.max(0, Math.min(1, Number(edge.norm_change || 0)));
      const peak = Number(edge.peak_time_index || 0);
      const distance = Math.abs(timeIndex - peak);
      const pulse = Math.max(0, 1 - (distance / 5));
      const alpha = Math.min(0.92, 0.08 + (norm * 0.24) + (pulse * 0.58));
      const widthPx = 0.8 + (norm * 2.2) + (pulse * 2.0);
      const [r, g, b] = directionColor(edge.direction);
      const sign = String(edge.src || "") <= String(edge.dst || "") ? 1 : -1;
      const mx = (srcPos.x + dstPos.x) / 2;
      const my = (srcPos.y + dstPos.y) / 2;
      const vx = dstPos.x - srcPos.x;
      const vy = dstPos.y - srcPos.y;
      const len = Math.max(1, Math.hypot(vx, vy));
      const nx = (-vy / len) * sign;
      const ny = (vx / len) * sign;
      const bend = 10 + (norm * 20);

      ctx.beginPath();
      ctx.moveTo(srcPos.x, srcPos.y);
      ctx.quadraticCurveTo(mx + (nx * bend), my + (ny * bend), dstPos.x, dstPos.y);
      ctx.lineWidth = widthPx;
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
      ctx.shadowBlur = 6 + (pulse * 14);
      ctx.shadowColor = `rgba(${r}, ${g}, ${b}, ${Math.min(0.8, alpha + 0.12)})`;
      ctx.stroke();
      ctx.shadowBlur = 0;

      if (pulse > 0.34) {
        activeEdgeRows.push({
          src: String(edge.src || ""),
          dst: String(edge.dst || ""),
          pulse,
          abs_change: Number(edge.abs_change || 0),
          direction: String(edge.direction || ""),
        });
      }
    });

    nodes.forEach((node) => {
      const id = String(node.id || "");
      const pos = positions[id];
      if (!pos) return;
      const strength = Number(nodeStrength[id] || 0);
      const between = Number(nodeBetweenness[id] || 0);
      const normalized = Math.max(0, Math.min(1, strength / Math.max(1e-9, Number(payload.maxStrength || 1))));
      const nodeRadius = 9 + (normalized * 15) + (Math.min(1, between) * 2.5);
      const color = roleColor(node.role);

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, nodeRadius + 5, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.05)";
      ctx.fill();

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, nodeRadius, 0, Math.PI * 2);
      const grad = ctx.createRadialGradient(pos.x - 2, pos.y - 3, 2, pos.x, pos.y, nodeRadius + 3);
      grad.addColorStop(0, "rgba(255,255,255,0.95)");
      grad.addColorStop(0.2, color);
      grad.addColorStop(1, "rgba(18, 28, 44, 0.96)");
      ctx.fillStyle = grad;
      ctx.strokeStyle = "rgba(210, 226, 252, 0.55)";
      ctx.lineWidth = 1.1;
      ctx.shadowBlur = 9;
      ctx.shadowColor = `${color}99`;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.stroke();

      ctx.fillStyle = "#f5f9ff";
      ctx.font = "600 11px IBM Plex Mono, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(id, pos.x, pos.y);
    });

    ctx.fillStyle = "rgba(214, 230, 252, 0.9)";
    ctx.font = "600 13px Space Grotesk, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(`t=${timeNorm.toFixed(3)}  •  frame ${safeIndex + 1}/${frames.length}`, 14, 22);

    if (UI.networkAnimMeta) {
      UI.networkAnimMeta.textContent = `t=${timeNorm.toFixed(3)} · frame ${safeIndex + 1}/${frames.length}`;
    }
    if (UI.networkAnimFrame && document.activeElement !== UI.networkAnimFrame) {
      UI.networkAnimFrame.value = String(safeIndex);
    }
    if (UI.networkAnimEdges) {
      const lines = activeEdgeRows
        .sort((a, b) => ((b.pulse * b.abs_change) - (a.pulse * a.abs_change)))
        .slice(0, 6)
        .map((row) => {
          const arrow = row.direction === "decrease" || row.direction === "down" ? "decreasing" : "increasing";
          return `${row.src} -> ${row.dst} (${arrow}; delta=${Number(row.abs_change || 0).toFixed(4)})`;
        });
      UI.networkAnimEdges.innerHTML = lines.length
        ? lines.map((line) => `<li>${escapeHtml(line)}</li>`).join("")
        : `<li class="muted">No strongly active edge transitions at this frame.</li>`;
    }
  }

  function startNetworkAnimationLoop() {
    stopNetworkAnimationLoop();
    if (!state.networkAnimPlaying || !state.networkAnimPayload) return;
    const payload = state.networkAnimPayload;
    const frames = Array.isArray(payload?.frames) ? payload.frames : [];
    if (frames.length <= 1) return;

    const stepMsBase = 320;
    const tick = (timestamp) => {
      if (!state.networkAnimPlaying || !state.networkAnimPayload) {
        state.networkAnimRafId = null;
        return;
      }
      if (!isDashboardTabActive() || document.hidden) {
        state.networkAnimLastTickMs = Number(timestamp || 0);
        state.networkAnimRafId = requestAnimationFrame(tick);
        return;
      }
      if (!state.networkAnimLastTickMs) {
        state.networkAnimLastTickMs = Number(timestamp || 0);
      }
      const elapsed = Number(timestamp || 0) - Number(state.networkAnimLastTickMs || 0);
      const speed = Math.max(0.5, Math.min(3, Number(state.networkAnimSpeed || 1)));
      const frameStepMs = stepMsBase / speed;
      if (elapsed >= frameStepMs) {
        state.networkAnimLastTickMs = Number(timestamp || 0);
        state.networkAnimFrameIndex = (Number(state.networkAnimFrameIndex || 0) + 1) % Math.max(1, frames.length);
        drawNetworkAnimationFrame(state.networkAnimPayload, state.networkAnimFrameIndex);
      }
      state.networkAnimRafId = requestAnimationFrame(tick);
    };
    state.networkAnimRafId = requestAnimationFrame(tick);
  }

  function bindNetworkAnimationControls() {
    if (UI.networkAnimToggle && !UI.networkAnimToggle.dataset.bound) {
      UI.networkAnimToggle.dataset.bound = "1";
      UI.networkAnimToggle.addEventListener("click", () => {
        state.networkAnimPlaying = !state.networkAnimPlaying;
        UI.networkAnimToggle.textContent = state.networkAnimPlaying ? "Pause" : "Play";
        if (state.networkAnimPlaying) startNetworkAnimationLoop();
        else stopNetworkAnimationLoop();
        updateNetworkAnimationStatus(state.networkAnimPayload);
      });
    }
    if (UI.networkAnimFrame && !UI.networkAnimFrame.dataset.bound) {
      UI.networkAnimFrame.dataset.bound = "1";
      UI.networkAnimFrame.addEventListener("input", () => {
        const idx = Number(UI.networkAnimFrame?.value || 0);
        state.networkAnimFrameIndex = Math.max(0, Math.round(idx));
        drawNetworkAnimationFrame(state.networkAnimPayload || { nodes: [], frames: [], edges: [], maxStrength: 1 }, state.networkAnimFrameIndex);
      });
      UI.networkAnimFrame.addEventListener("change", () => {
        drawNetworkAnimationFrame(state.networkAnimPayload || { nodes: [], frames: [], edges: [], maxStrength: 1 }, state.networkAnimFrameIndex);
      });
    }
    if (UI.networkAnimSpeed && !UI.networkAnimSpeed.dataset.bound) {
      UI.networkAnimSpeed.dataset.bound = "1";
      UI.networkAnimSpeed.addEventListener("input", () => {
        state.networkAnimSpeed = Math.max(0.5, Math.min(3, Number(UI.networkAnimSpeed?.value || 1)));
      });
    }
    if (!window.__phoenixNetworkAnimResizeBound) {
      window.__phoenixNetworkAnimResizeBound = true;
      window.addEventListener("resize", () => {
        state.networkAnimPositions = {};
        if (state.networkAnimPayload) {
          drawNetworkAnimationFrame(state.networkAnimPayload, state.networkAnimFrameIndex);
        }
      });
    }
    if (!window.__phoenixNetworkAnimTabBound) {
      window.__phoenixNetworkAnimTabBound = true;
      window.addEventListener("phoenix:tab-switch", (event) => {
        const tabId = String(event?.detail?.tabId || "");
        if (tabId === "dashboard") {
          if (state.networkAnimPayload) {
            drawNetworkAnimationFrame(state.networkAnimPayload, state.networkAnimFrameIndex);
            if (state.networkAnimPlaying && !state.networkAnimRafId) {
              startNetworkAnimationLoop();
            }
          }
          return;
        }
        stopNetworkAnimationLoop();
      });
    }
  }

  function renderTimeVaryingNetwork(dashboard) {
    if (!UI.networkAnimCanvas) return;
    bindNetworkAnimationControls();
    const raw = dashboard?.network?.time_varying_network || {};
    const payload = normalizeNetworkPayload(raw);
    const key = networkPayloadIdentity(payload);

    const ready = payload.status === "ready" && payload.frames.length > 0 && payload.nodes.length > 0;
    if (!ready) {
      stopNetworkAnimationLoop();
      state.networkAnimPayload = null;
      state.networkAnimPayloadKey = "";
      state.networkAnimFrameIndex = 0;
      setNetworkAnimStatus("UNAVAILABLE", "idle");
      if (UI.networkAnimCaption) {
        UI.networkAnimCaption.textContent = payload.reason || "No time-varying network artifacts available for this run.";
      }
      if (UI.networkAnimEdges) {
        UI.networkAnimEdges.innerHTML = `<li class="muted">Run a cycle with temporal network outputs to animate dynamics.</li>`;
      }
      if (UI.networkAnimMeta) UI.networkAnimMeta.textContent = "t=0.00 · frame 0/0";
      if (UI.networkAnimFrame) {
        UI.networkAnimFrame.min = "0";
        UI.networkAnimFrame.max = "0";
        UI.networkAnimFrame.value = "0";
        UI.networkAnimFrame.disabled = true;
      }
      if (UI.networkAnimSpeed) {
        UI.networkAnimSpeed.disabled = true;
      }
      if (UI.networkAnimToggle) {
        UI.networkAnimToggle.disabled = true;
        UI.networkAnimToggle.textContent = "Play";
      }
      drawNetworkAnimationFrame({ nodes: [], frames: [], edges: [], maxStrength: 1 }, 0);
      return;
    }

    state.networkAnimPayload = payload;
    const payloadChanged = state.networkAnimPayloadKey !== key;
    if (payloadChanged) {
      state.networkAnimPayloadKey = key;
      state.networkAnimFrameIndex = 0;
      state.networkAnimPositions = {};
      state.networkAnimLastTickMs = 0;
    } else {
      state.networkAnimFrameIndex = Math.max(
        0,
        Math.min(payload.frames.length - 1, Number(state.networkAnimFrameIndex || 0)),
      );
    }

    if (UI.networkAnimFrame) {
      UI.networkAnimFrame.min = "0";
      UI.networkAnimFrame.max = String(Math.max(0, payload.frames.length - 1));
      UI.networkAnimFrame.value = String(state.networkAnimFrameIndex);
      UI.networkAnimFrame.disabled = payload.frames.length <= 1;
    }
    if (UI.networkAnimSpeed) {
      UI.networkAnimSpeed.value = String(state.networkAnimSpeed || 1);
      UI.networkAnimSpeed.disabled = payload.frames.length <= 1;
    }
    if (UI.networkAnimToggle) {
      UI.networkAnimToggle.disabled = payload.frames.length <= 1;
      UI.networkAnimToggle.textContent = payload.frames.length <= 1 ? "Static" : (state.networkAnimPlaying ? "Pause" : "Play");
    }
    if (UI.networkAnimCaption) {
      const summary = payload.summary || {};
      const base = `Animated from PHOENIX network artifacts: ${summary.node_count || payload.nodes.length} nodes, ${summary.edge_count || payload.edges.length} dynamic edges, ${summary.frame_count || payload.frames.length} frames.`;
      UI.networkAnimCaption.textContent = payload.frames.length <= 1
        ? `${base} This run produced a single temporal frame, so playback is static.`
        : base;
    }
    updateNetworkAnimationStatus(payload);
    drawNetworkAnimationFrame(payload, state.networkAnimFrameIndex);
    if (state.networkAnimPlaying && payload.frames.length > 1) {
      if (payloadChanged) {
        stopNetworkAnimationLoop();
        startNetworkAnimationLoop();
      } else if (!state.networkAnimRafId) {
        startNetworkAnimationLoop();
      }
    } else {
      stopNetworkAnimationLoop();
    }
  }

  function buildBarChart(canvasId, labels, values, color, opts = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === "undefined") return null;
    const hasExplicitMax = Object.prototype.hasOwnProperty.call(opts, "maxY");
    const maxY = hasExplicitMax ? opts.maxY : 1;
    return new Chart(canvas, {
      type: "bar",
      data: {
        labels,
        datasets: [{ label: "score", data: values, backgroundColor: color, borderRadius: 4 }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: {
            beginAtZero: true,
            ...(maxY != null ? { max: Number(maxY) } : {}),
            ticks: { color: "#b7cae8" },
            grid: { color: "rgba(109, 136, 180, 0.2)" },
          },
          x: {
            ticks: { color: "#b7cae8" },
            grid: { color: "rgba(109, 136, 180, 0.1)" },
          },
        },
      },
    });
  }

  function buildMultiBarChart(canvasId, labels, datasets, opts = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === "undefined") return null;
    const maxY = Number.isFinite(Number(opts.maxY)) ? Number(opts.maxY) : null;
    const stacked = Boolean(opts.stacked);
    return new Chart(canvas, {
      type: "bar",
      data: {
        labels,
        datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, labels: { color: "#c6d8f3", boxWidth: 12 } },
        },
        scales: {
          y: {
            beginAtZero: true,
            ...(maxY != null ? { max: maxY } : {}),
            stacked,
            ticks: { color: "#b7cae8" },
            grid: { color: "rgba(109, 136, 180, 0.18)" },
          },
          x: {
            stacked,
            ticks: { color: "#b7cae8", maxRotation: 0, autoSkip: true, maxTicksLimit: 12 },
            grid: { color: "rgba(109, 136, 180, 0.10)" },
          },
        },
      },
    });
  }

  function buildLineChart(canvasId, labels, datasets, opts = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === "undefined") return null;
    const maxY = Number.isFinite(Number(opts.maxY)) ? Number(opts.maxY) : null;
    const minY = Number.isFinite(Number(opts.minY)) ? Number(opts.minY) : null;
    return new Chart(canvas, {
      type: "line",
      data: {
        labels,
        datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, labels: { color: "#c6d8f3", boxWidth: 12 } },
        },
        scales: {
          y: {
            beginAtZero: minY == null ? true : false,
            ...(minY != null ? { min: minY } : {}),
            ...(maxY != null ? { max: maxY } : {}),
            ticks: { color: "#b7cae8" },
            grid: { color: "rgba(109, 136, 180, 0.18)" },
          },
          x: {
            ticks: { color: "#b7cae8", maxTicksLimit: 12 },
            grid: { color: "rgba(109, 136, 180, 0.10)" },
          },
        },
      },
    });
  }

  function parseCsvText(raw) {
    const lines = String(raw || "")
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0);
    if (!lines.length) return { headers: [], rows: [] };
    const headers = lines[0].split(",").map((cell) => cell.trim());
    const rows = lines.slice(1).map((line) => {
      const cells = line.split(",");
      const row = {};
      headers.forEach((key, idx) => {
        row[key] = String(cells[idx] ?? "").trim();
      });
      return row;
    });
    return { headers, rows };
  }

  function parseNumeric(value) {
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
  }

  function mean(values) {
    const xs = (values || []).filter((v) => Number.isFinite(v));
    if (!xs.length) return null;
    return xs.reduce((a, b) => a + b, 0) / xs.length;
  }

  function stdev(values) {
    const xs = (values || []).filter((v) => Number.isFinite(v));
    if (xs.length < 2) return 0;
    const mu = mean(xs) || 0;
    const variance = xs.reduce((acc, v) => acc + ((v - mu) ** 2), 0) / (xs.length - 1);
    return Math.sqrt(Math.max(0, variance));
  }

  const SERIES_COLORS = [
    "#4f8dff",
    "#11c4a5",
    "#f59f3a",
    "#e4588f",
    "#8d6bff",
    "#18b6d9",
    "#f06d5f",
    "#6ccf5f",
    "#f2c84b",
    "#4bc0ff",
    "#ff7ab6",
    "#a5b4fc",
  ];

  function colorForSeries(index, alpha = 1) {
    const idx = Math.abs(Number(index) || 0) % SERIES_COLORS.length;
    const hex = SERIES_COLORS[idx];
    if (alpha >= 1) return hex;
    const rgb = hex.replace("#", "");
    const r = parseInt(rgb.slice(0, 2), 16);
    const g = parseInt(rgb.slice(2, 4), 16);
    const b = parseInt(rgb.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, alpha))})`;
  }

  function buildVarLabelMap(snapshot) {
    const map = {};
    const schemaVars = snapshot?.collection_schema?.variables || [];
    schemaVars.forEach((row) => {
      const varId = String(row?.var_id || "").trim();
      if (!varId) return;
      const label = String(row?.label || "").trim();
      if (label) map[varId] = label;
    });
    const opVars = snapshot?.operationalization_summary?.variables || [];
    opVars.forEach((row) => {
      const varId = String(row?.var_id || "").trim();
      if (!varId || map[varId]) return;
      const label = String(row?.label || "").trim();
      if (label) map[varId] = label;
    });
    return map;
  }

  function toDisplayLabel(varId, labelMap) {
    const token = String(varId || "").trim();
    const label = String((labelMap || {})[token] || token).trim();
    if (!token) return label || "unknown";
    if (!label || label === token) return token;
    return `${label} (${token})`;
  }

  function normalizeSeries(values) {
    const ys = Array.isArray(values) ? values : [];
    const finite = ys.filter((v) => Number.isFinite(v));
    if (!finite.length) return ys.map(() => null);
    const minVal = Math.min(...finite);
    const maxVal = Math.max(...finite);
    const span = Math.max(1e-9, maxVal - minVal);
    return ys.map((v) => (Number.isFinite(v) ? (v - minVal) / span : null));
  }

  async function ensureTimeSeriesData(snapshot) {
    const profileId = String(snapshot?.session?.profile_id || "").trim();
    if (!profileId || !snapshot?.has_pseudodata) {
      state.timeSeriesData = null;
      state.timeSeriesCacheKey = "";
      return null;
    }
    const cacheKey = [
      profileId,
      String(snapshot?.session?.updated_at || ""),
      String(snapshot?.pseudodata_summary?.n_points || ""),
    ].join("|");
    if (state.timeSeriesData && state.timeSeriesCacheKey === cacheKey) {
      return state.timeSeriesData;
    }

    const relPath = `outputs/pseudodata/${profileId}/pseudodata_wide.csv`;
    const response = await fetch(`/api/sessions/${sessionId}/files/${encodeURI(relPath)}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch pseudodata CSV (${response.status}).`);
    }
    const text = await response.text();
    const parsed = parseCsvText(text);
    if (!parsed.headers.length) {
      state.timeSeriesData = null;
      state.timeSeriesCacheKey = cacheKey;
      return null;
    }

    const variableColumns = parsed.headers.filter((name) => !["t_index", "date"].includes(String(name || "").trim()));
    const labelMap = buildVarLabelMap(snapshot);
    const x = parsed.rows.map((row, idx) => {
      const tVal = parseNumeric(row.t_index);
      if (tVal != null) return tVal;
      return idx + 1;
    });
    const dates = parsed.rows.map((row) => String(row.date || "").trim());
    const series = {};
    const stats = {};
    variableColumns.forEach((col) => {
      const values = parsed.rows.map((row) => parseNumeric(row[col]));
      const finite = values.filter((v) => Number.isFinite(v));
      series[col] = values;
      stats[col] = {
        mean: mean(finite),
        std: stdev(finite),
        missing: values.filter((v) => v == null).length,
        n: values.length,
      };
    });

    state.timeSeriesData = {
      x,
      dates,
      series,
      stats,
      columns: variableColumns,
      labelMap,
      rowCount: parsed.rows.length,
    };
    state.timeSeriesCacheKey = cacheKey;
    return state.timeSeriesData;
  }

  function ensureTimeSeriesSelectors(data) {
    const variableSelect = document.getElementById("timeseries-variable-select");
    if (!variableSelect) return;
    const columns = Array.isArray(data?.columns) ? data.columns : [];
    if (!columns.length) {
      variableSelect.innerHTML = `<option value="">No signals</option>`;
      variableSelect.disabled = true;
      return;
    }
    variableSelect.disabled = false;
    const current = String(variableSelect.value || "");
    const labelMap = data?.labelMap || {};
    const rows = [`<option value="__all__">All variables</option>`];
    columns.forEach((name) => {
      rows.push(`<option value="${escapeHtml(name)}">${escapeHtml(toDisplayLabel(name, labelMap))}</option>`);
    });
    variableSelect.innerHTML = rows.join("");
    if (current && columns.includes(current)) {
      variableSelect.value = current;
    } else if (current === "__all__") {
      variableSelect.value = "__all__";
    } else {
      variableSelect.value = "__all__";
    }
  }

  function updateTimeSeriesStatus(text) {
    const node = document.getElementById("timeseries-status");
    if (node) node.textContent = String(text || "");
  }

  function renderTimeSeriesDiagnostics(data, signalColumns, labelMap) {
    const node = document.getElementById("timeseries-signal-diagnostics");
    if (!node) return;
    const cols = Array.isArray(signalColumns) ? signalColumns : [];
    if (!cols.length) {
      node.innerHTML = `<li class="muted">No signal diagnostics available.</li>`;
      return;
    }
    const statsMap = data?.stats || {};
    node.innerHTML = cols.slice(0, 10).map((col) => {
      const stats = statsMap[col] || {};
      const idx = Math.max(0, (data?.columns || []).indexOf(col));
      const color = colorForSeries(idx, 1);
      const label = toDisplayLabel(col, labelMap);
      const meanVal = Number.isFinite(stats.mean) ? Number(stats.mean).toFixed(2) : "n/a";
      const stdVal = Number.isFinite(stats.std) ? Number(stats.std).toFixed(2) : "n/a";
      const missingPct = Number(stats.n || 0) > 0
        ? ((Number(stats.missing || 0) / Number(stats.n || 1)) * 100).toFixed(1)
        : "0.0";
      return `
        <li>
          <span class="series-color-dot" style="background:${escapeHtml(color)}"></span>
          ${escapeHtml(label)} · mean=${escapeHtml(meanVal)} · std=${escapeHtml(stdVal)} · missing=${escapeHtml(missingPct)}%
        </li>
      `;
    }).join("");
  }

  function renderTimeSeriesChart() {
    const canvas = document.getElementById("chart-timeseries");
    const variableSelect = document.getElementById("timeseries-variable-select");
    const windowSelect = document.getElementById("timeseries-window-select");
    if (!canvas || !variableSelect || !windowSelect || typeof Chart === "undefined") return;
    const data = state.timeSeriesData;
    if (!data || !Array.isArray(data.columns) || !data.columns.length) {
      if (state.timeSeriesChart && typeof state.timeSeriesChart.destroy === "function") {
        state.timeSeriesChart.destroy();
      }
      state.timeSeriesChart = null;
      updateTimeSeriesStatus("No pseudodata loaded.");
      renderTimeSeriesDiagnostics(null, [], {});
      return;
    }

    const signal = String(variableSelect.value || data.columns[0] || "");
    const normalizeToggle = Boolean(document.getElementById("timeseries-normalize")?.checked);
    if (!signal || (signal !== "__all__" && !data.series[signal])) {
      updateTimeSeriesStatus("Select a signal to display time-series data.");
      renderTimeSeriesDiagnostics(data, [], data.labelMap || {});
      return;
    }

    const windowRaw = String(windowSelect.value || "all");
    const requestedWindow = windowRaw === "all" ? data.x.length : Math.max(1, Number(windowRaw) || data.x.length);
    const start = Math.max(0, data.x.length - requestedWindow);
    const xSlice = data.x.slice(start);
    const dateSlice = data.dates.slice(start);
    const labelMap = data.labelMap || {};
    const signalColumns = signal === "__all__" ? data.columns : [signal];
    const datasets = signalColumns.map((column) => {
      const rawSlice = (data.series[column] || []).slice(start);
      const ySlice = normalizeToggle ? normalizeSeries(rawSlice) : rawSlice;
      const baseIndex = Math.max(0, data.columns.indexOf(column));
      const color = colorForSeries(baseIndex, 1);
      return {
        label: toDisplayLabel(column, labelMap),
        data: ySlice,
        borderColor: color,
        backgroundColor: colorForSeries(baseIndex, signal === "__all__" ? 0.08 : 0.16),
        tension: 0.2,
        pointRadius: signal === "__all__" ? 1.2 : 1.8,
        pointHoverRadius: signal === "__all__" ? 2.2 : 3.2,
        borderWidth: signal === "__all__" ? 1.8 : 2.2,
        spanGaps: true,
      };
    });

    if (state.timeSeriesChart && typeof state.timeSeriesChart.destroy === "function") {
      state.timeSeriesChart.destroy();
    }
    state.timeSeriesChart = new Chart(canvas, {
      type: "line",
      data: {
        labels: xSlice,
        datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, labels: { color: "#c6d8f3", boxWidth: 12 } },
          tooltip: {
            callbacks: {
              title: (ctx) => {
                const idx = Number(ctx?.[0]?.dataIndex ?? -1);
                const date = idx >= 0 ? String(dateSlice[idx] || "") : "";
                const t = idx >= 0 ? String(xSlice[idx] || "") : "";
                return date ? `t=${t} (${date})` : `t=${t}`;
              },
            },
          },
        },
        scales: {
          y: {
            ticks: { color: "#b7cae8" },
            grid: { color: "rgba(109, 136, 180, 0.18)" },
          },
          x: {
            ticks: { color: "#b7cae8", maxTicksLimit: 12 },
            grid: { color: "rgba(109, 136, 180, 0.1)" },
          },
        },
      },
    });
    if (signal === "__all__") {
      const aggregateMissing = signalColumns.reduce((acc, col) => {
        const ys = (data.series[col] || []).slice(start);
        return acc + ys.filter((v) => v == null).length;
      }, 0);
      updateTimeSeriesStatus(
        `Signals=${signalColumns.length} | points=${xSlice.length} / ${data.x.length} | missing cells=${aggregateMissing}${normalizeToggle ? " | normalized=on" : ""}`,
      );
    } else {
      const ySlice = (data.series[signal] || []).slice(start);
      updateTimeSeriesStatus(
        `Signal=${toDisplayLabel(signal, labelMap)} | points=${xSlice.length} / ${data.x.length} | missing=${ySlice.filter((v) => v == null).length}${normalizeToggle ? " | normalized=on" : ""}`,
      );
    }
    renderTimeSeriesDiagnostics(data, signalColumns, labelMap);
  }

  async function refreshTimeSeries(snapshot) {
    const variableSelect = document.getElementById("timeseries-variable-select");
    if (!variableSelect) return;
    if (!snapshot?.has_pseudodata) {
      state.timeSeriesData = null;
      state.timeSeriesCacheKey = "";
      variableSelect.innerHTML = `<option value=\"\">No signals</option>`;
      variableSelect.disabled = true;
      renderTimeSeriesChart();
      return;
    }
    try {
      const data = await ensureTimeSeriesData(snapshot);
      ensureTimeSeriesSelectors(data);
      renderTimeSeriesChart();
    } catch (error) {
      updateTimeSeriesStatus(`Time-series load failed: ${error.message || error}`);
    }
  }

  function renderMetricCards(dashboard) {
    const readiness = dashboard.readiness || {};
    const impact = dashboard.impact || {};
    const step05 = dashboard.step05 || {};
    const topPredictor = (impact.top_predictors || [])[0] || {};
    const labelMap = buildVarLabelMap(state.snapshot || {});
    const topPredictorName = topPredictor.predictor_display
      || topPredictor.predictor_label
      || toDisplayLabel(topPredictor.predictor || "", labelMap)
      || "—";

    const setText = (id, value) => {
      const node = document.getElementById(id);
      if (node) node.textContent = value;
    };
    setText("metric-readiness-score", `${Number(readiness.score_0_100 || 0).toFixed(1)}`);
    setText("metric-readiness-label", readiness.label || "unknown");
    setText("metric-tier", readiness.tier || "—");
    setText("metric-variant", readiness.tier3_variant || "—");
    setText("metric-top-predictor", topPredictorName);
    setText(
      "metric-top-predictor-score",
      topPredictor.score_0_1 != null ? Number(topPredictor.score_0_1).toFixed(3) : "—",
    );
    setText("metric-barrier-count", String((step05.selected_barriers || []).length || 0));
  }

  function renderDashboard(dashboard) {
    renderMetricCards(dashboard);
    destroyCharts();
    const labelMap = buildVarLabelMap(state.snapshot || {});

    const impactRows = (dashboard.impact?.top_predictors || []).slice(0, 8);
    const barrierRows = (dashboard.step05?.selected_barriers || []).slice(0, 8);
    const copingRows = (dashboard.step05?.selected_coping || []).slice(0, 8);

    state.impactChart = buildBarChart(
      "chart-impact",
      impactRows.map((x) => x.predictor_display || x.predictor_label || toDisplayLabel(x.predictor || "", labelMap) || "n/a"),
      impactRows.map((x) => Number(x.score_0_1 || 0)),
      "#4f8dff",
    );
    state.barrierChart = buildBarChart(
      "chart-barriers",
      barrierRows.map((x) => x.barrier_name || "n/a"),
      barrierRows.map((x) => Number(x.score_0_1 || 0)),
      "#f4b43b",
    );
    state.copingChart = buildBarChart(
      "chart-coping",
      copingRows.map((x) => x.coping_name || "n/a"),
      copingRows.map((x) => Number(x.score_0_1 || 0)),
      "#28c995",
    );
    renderTimeVaryingNetwork(dashboard);

    const renderList = (id, rows, emptyText) => {
      const node = document.getElementById(id);
      if (!node) return;
      const list = Array.isArray(rows) ? rows : [];
      node.innerHTML = list.length
        ? list.map((item) => `<li>${escapeHtml(item)}</li>`).join("")
        : `<li class="muted">${escapeHtml(emptyText)}</li>`;
    };

    renderList("dashboard-readiness-why", dashboard.readiness?.why || [], "No readiness rationale available.");
    const networkNotes = dashboard.network?.notes || [];
    renderList("dashboard-network-notes", networkNotes, "No network notes available.");
    const targetRows = (dashboard.step03?.recommended_targets || []).map((row) => {
      const p = row.predictor_label || toDisplayLabel(row.predictor || "", labelMap) || "unknown";
      const s = Number(row.score_0_1 || 0).toFixed(3);
      const r = row.rationale ? ` — ${row.rationale}` : "";
      return `${p} (score=${s})${r}`;
    });
    const step03Empty = dashboard.step03?.status === "skipped"
      ? "Step-03 was skipped (no impact output available)."
      : "No Step-03 targets available.";
    renderList("dashboard-step03-targets", targetRows, step03Empty);

    const step04 = dashboard.step04 || {};
    const step04Rows = [];
    const predictorRows = Array.isArray(step04.recommended_predictor_rows)
      ? step04.recommended_predictor_rows
      : [];
    if (predictorRows.length) {
      const predictorText = predictorRows
        .map((row) => toDisplayLabel(row?.predictor || "", { ...(labelMap || {}), [String(row?.predictor || "")]: String(row?.label || "") }))
        .join(", ");
      step04Rows.push(`Predictors (${predictorRows.length}): ${predictorText}`);
    } else if ((step04.recommended_predictors || []).length) {
      const predictorText = (step04.recommended_predictors || []).map((id) => toDisplayLabel(id, labelMap)).join(", ");
      step04Rows.push(`Predictors (${step04.recommended_predictors.length}): ${predictorText}`);
    }
    if ((step04.retained_criteria_ids || []).length) {
      const retained = (step04.retained_criteria_ids || []).map((id) => toDisplayLabel(id, labelMap)).join(", ");
      step04Rows.push(`Retained criteria (${step04.retained_criteria_ids.length}): ${retained}`);
    }
    if ((step04.reason_codes || []).length) {
      step04Rows.push(`Range-policy reasons: ${step04.reason_codes.join(", ")}`);
    }
    const step04Empty = step04.status === "skipped"
      ? "Step-04 was skipped because upstream target selection was unavailable."
      : "No Step-04 updated model details available.";
    renderList("dashboard-step04-updated", step04Rows, step04Empty);

    const summary = String(dashboard.step05?.user_summary || "").trim();
    const step05SummaryNode = document.getElementById("dashboard-step05-summary");
    if (step05SummaryNode) {
      if (summary) {
        step05SummaryNode.textContent = summary;
      } else if (dashboard.step05?.status === "skipped") {
        step05SummaryNode.textContent = "Step-05 intervention was skipped because no handoff outputs were available.";
      } else {
        step05SummaryNode.textContent = "No intervention summary yet.";
      }
    }

    /* HAPA Intervention Message — prominent panel in Dashboard */
    const interventionPanel = document.getElementById("panel-intervention-message");
    const interventionText = document.getElementById("dashboard-intervention-message");
    const interventionStructured = document.getElementById("dashboard-intervention-structure");
    if (interventionPanel && interventionText) {
      if (summary) {
        interventionText.textContent = summary;
        const selectedTargets = (dashboard.step05?.selected_targets || []).slice(0, 5);
        const selectedBarriers = (dashboard.step05?.selected_barriers || []).slice(0, 6);
        const selectedCoping = (dashboard.step05?.selected_coping || []).slice(0, 6);
        const hapaPlan = (dashboard.step05?.hapa_component_plan || []).slice(0, 4);
        const phasedPlan = (dashboard.step05?.phased_delivery_plan || []).slice(0, 3);
        const monitoring = (dashboard.step05?.monitoring_plan || []).slice(0, 5);
        const safety = (dashboard.step05?.safety_notes || []).slice(0, 4);

        const mkList = (rows, fn, emptyText) => (
          Array.isArray(rows) && rows.length
            ? `<ul>${rows.map((item) => `<li>${fn(item)}</li>`).join("")}</ul>`
            : `<p class="muted compact">${escapeHtml(emptyText)}</p>`
        );

        if (interventionStructured) {
          interventionStructured.innerHTML = `
            <div class="detail-grid" style="margin-top:10px">
              <article class="detail-card">
                <h3>Selected Treatment Targets</h3>
                ${mkList(
                  selectedTargets,
                  (item) => `${escapeHtml(String(item.predictor_label || item.predictor || "target"))} (priority=${Number(item.priority_0_1 || 0).toFixed(3)})`,
                  "No treatment targets reported.",
                )}
              </article>
              <article class="detail-card">
                <h3>Top Barriers</h3>
                ${mkList(
                  selectedBarriers,
                  (item) => `${escapeHtml(String(item.barrier_name || "barrier"))} (score=${Number(item.score_0_1 || 0).toFixed(3)})`,
                  "No barriers reported.",
                )}
              </article>
              <article class="detail-card">
                <h3>Top Coping Strategies</h3>
                ${mkList(
                  selectedCoping,
                  (item) => `${escapeHtml(String(item.coping_name || "strategy"))} (score=${Number(item.score_0_1 || 0).toFixed(3)})`,
                  "No coping strategies reported.",
                )}
              </article>
              <article class="detail-card">
                <h3>HAPA Components</h3>
                ${mkList(
                  hapaPlan,
                  (item) => `<strong>${escapeHtml(String(item.component || "component"))}</strong>: ${escapeHtml(String(item.objective || ""))}`,
                  "No HAPA component plan reported.",
                )}
              </article>
              <article class="detail-card">
                <h3>Phased Delivery</h3>
                ${mkList(
                  phasedPlan,
                  (item) => `<strong>${escapeHtml(String(item.phase || "phase"))}</strong> (${escapeHtml(String(item.time_window || ""))}): ${escapeHtml(String(item.primary_goal || ""))}`,
                  "No phased delivery plan reported.",
                )}
              </article>
              <article class="detail-card">
                <h3>Monitoring and Safety</h3>
                ${mkList(
                  [...monitoring, ...safety],
                  (item) => escapeHtml(String(item || "")),
                  "No monitoring/safety notes reported.",
                )}
              </article>
            </div>
          `;
        }
        interventionPanel.style.display = "block";
      } else {
        if (interventionStructured) interventionStructured.innerHTML = "";
        interventionPanel.style.display = "none";
      }
    }

    /* Static image galleries are intentionally hidden in frontend; dashboard uses computed charts instead. */
    const cycleVisualsPanel = document.getElementById("panel-cycle-visuals");
    if (cycleVisualsPanel) cycleVisualsPanel.style.display = "none";

    if (dashboard.impact?.status === "skipped") {
      const reason = String(dashboard.impact?.status_reason || "").trim();
      const noteNode = document.getElementById("dashboard-network-notes");
      if (noteNode && reason) {
        const exists = Array.from(noteNode.querySelectorAll("li")).some((li) => li.textContent === reason);
        if (!exists) {
          noteNode.insertAdjacentHTML("beforeend", `<li>${escapeHtml(reason)}</li>`);
        }
      }
    }
    renderIntermediatePanels(dashboard);
  }

  function renderIntermediatePanels(dashboard) {
    const snapshot = state.snapshot || {};
    const session = snapshot.session || {};
    const notes = session.notes || {};
    const labelMap = buildVarLabelMap(snapshot);
    const readiness = dashboard.readiness || {};
    const network = dashboard.network || {};
    const impact = dashboard.impact || {};
    const runtime = dashboard.runtime || {};

    const readinessNode = document.getElementById("dashboard-readiness-overview");
    if (readinessNode) {
      const ds = readiness.dataset_overview || {};
      readinessNode.innerHTML = `
        <div class="meta-grid">
          <div><strong>Rows:</strong> ${escapeHtml(String(ds.n_rows ?? "—"))}</div>
          <div><strong>Candidate variables:</strong> ${escapeHtml(String(ds.candidate_cols_count ?? "—"))}</div>
          <div><strong>Correlation method:</strong> ${escapeHtml(String(ds.corr_method || "—"))}</div>
        </div>
      `;
    }

    const readinessVarNode = document.getElementById("dashboard-readiness-variables");
    if (readinessVarNode) {
      const rows = Array.isArray(readiness.variable_quality_rows) ? readiness.variable_quality_rows.slice(0, 8) : [];
      if (!rows.length) {
        readinessVarNode.innerHTML = `<li class="muted">No variable diagnostics available.</li>`;
      } else {
        readinessVarNode.innerHTML = rows.map((row) => {
          const name = toDisplayLabel(row.var_id || "", { ...(labelMap || {}), [String(row.var_id || "")]: String(row.label || "") });
          const missing = Number(row.missing_pct || 0).toFixed(1);
          const ac = Number(row.autocorr_lag1 || 0).toFixed(2);
          return `<li>${escapeHtml(name)} · missing=${missing}% · autocorr=${ac}</li>`;
        }).join("");
      }
    }

    const componentEntries = Object.entries(readiness.score_components || {})
      .map(([k, v]) => ({ key: k, label: String(k || "").replaceAll("_", " "), value: Number(v || 0) }))
      .sort((a, b) => b.value - a.value);
    state.readinessComponentsChart = buildBarChart(
      "chart-readiness-components",
      componentEntries.map((x) => x.label),
      componentEntries.map((x) => Number(x.value || 0) / 100),
      "#11c4a5",
    );

    const networkOverviewNode = document.getElementById("dashboard-network-overview");
    if (networkOverviewNode) {
      const methodStatus = network.method_status || {};
      const statuses = Object.keys(methodStatus).map((k) => `${k}:${methodStatus[k]}`).join(" | ") || "n/a";
      networkOverviewNode.innerHTML = `
        <div class="meta-grid">
          <div><strong>Analysis set:</strong> ${escapeHtml(String(network.analysis_set || "—"))}</div>
          <div><strong>Method status:</strong> ${escapeHtml(statuses)}</div>
        </div>
      `;
    }

    const methodRows = Array.isArray(network.method_prediction_rows) ? network.method_prediction_rows : [];
    state.networkMethodChart = buildMultiBarChart(
      "chart-network-methods",
      methodRows.map((x) => String(x.method || "method")),
      [
        {
          label: "r2_overall",
          data: methodRows.map((x) => Number(x.r2_overall || 0)),
          backgroundColor: "#4f8dff",
        },
        {
          label: "aux_strength",
          data: methodRows.map((x) => Number(x.aux_strength || 0)),
          backgroundColor: "#f59f3a",
        },
      ],
      { maxY: 1 },
    );

    const temporalRows = Array.isArray(network.temporal_metrics) ? network.temporal_metrics : [];
    state.networkTemporalTrendChart = buildLineChart(
      "chart-network-temporal",
      temporalRows.map((x) => `t${x.time_index}`),
      [
        {
          label: "global_efficiency",
          data: temporalRows.map((x) => Number(x.global_efficiency || 0)),
          borderColor: "#18b6d9",
          backgroundColor: "rgba(24, 182, 217, 0.12)",
          tension: 0.2,
          pointRadius: 1.6,
          borderWidth: 2,
        },
        {
          label: "density",
          data: temporalRows.map((x) => Number(x.density || 0)),
          borderColor: "#8d6bff",
          backgroundColor: "rgba(141, 107, 255, 0.12)",
          tension: 0.2,
          pointRadius: 1.6,
          borderWidth: 2,
        },
        {
          label: "modularity",
          data: temporalRows.map((x) => Number(x.modularity || 0)),
          borderColor: "#f06d5f",
          backgroundColor: "rgba(240, 109, 95, 0.12)",
          tension: 0.2,
          pointRadius: 1.6,
          borderWidth: 2,
        },
      ],
      { minY: -0.1, maxY: 1 },
    );

    const predictorRows = Array.isArray(network.predictor_importance_rows) ? network.predictor_importance_rows.slice(0, 8) : [];
    state.networkPredictorImportanceChart = buildMultiBarChart(
      "chart-network-predictor-importance",
      predictorRows.map((x) => toDisplayLabel(x.predictor || "", { ...(labelMap || {}), [String(x.predictor || "")]: String(x.predictor_label || "") })),
      [
        {
          label: "network score",
          data: predictorRows.map((x) => Number(x.score || 0)),
          backgroundColor: "#6ccf5f",
        },
        {
          label: "delta_mse_criteria",
          data: predictorRows.map((x) => Number(x.delta_mse_criteria || 0)),
          backgroundColor: "#4bc0ff",
        },
      ],
      {},
    );

    const impactRows = Array.isArray(impact.decomposition_rows) ? impact.decomposition_rows.slice(0, 6) : [];
    state.impactDecompositionChart = buildMultiBarChart(
      "chart-impact-decomposition",
      impactRows.map((x) => x.predictor_display || x.predictor_label || x.predictor || "predictor"),
      [
        {
          label: "edge component",
          data: impactRows.map((x) => Number(x.edge_component || 0)),
          backgroundColor: "#4f8dff",
        },
        {
          label: "delta component",
          data: impactRows.map((x) => Number(x.delta_component || 0)),
          backgroundColor: "#f4b43b",
        },
        {
          label: "local component",
          data: impactRows.map((x) => Number(x.local_component || 0)),
          backgroundColor: "#11c4a5",
        },
      ],
      { stacked: true },
    );

    const stageRowsFromRuntime = Array.isArray(runtime.stage_durations_seconds) ? runtime.stage_durations_seconds : [];
    const stageRows = stageRowsFromRuntime.length
      ? stageRowsFromRuntime
      : Array.isArray(snapshot.pipeline_summary?.stage_results)
        ? snapshot.pipeline_summary.stage_results.map((row) => ({
          stage: row.stage,
          duration_seconds: Number(row.duration_seconds || 0),
          return_code: Number(row.return_code || 0),
        }))
        : [];
    state.stageRuntimeChart = buildBarChart(
      "chart-stage-runtime",
      stageRows.map((x) => String(x.stage || "")),
      stageRows.map((x) => Number(x.duration_seconds || 0)),
      "#a5b4fc",
      { maxY: null },
    );

    const stageCounts = runtime.stage_status_counts && typeof runtime.stage_status_counts === "object"
      ? runtime.stage_status_counts
      : {};
    const stageCountRows = [
      { key: "succeeded", label: "Succeeded", value: Number(stageCounts.succeeded || 0), color: "#34d399" },
      { key: "running", label: "Running", value: Number(stageCounts.running || 0), color: "#4f8dff" },
      { key: "failed", label: "Failed", value: Number(stageCounts.failed || 0), color: "#f87171" },
      { key: "skipped", label: "Skipped", value: Number(stageCounts.skipped || 0), color: "#fbbf24" },
      { key: "unknown", label: "Unknown", value: Number(stageCounts.unknown || 0), color: "#9cb2d3" },
    ].filter((row) => row.value > 0);
    state.stageStatusChart = buildBarChart(
      "chart-stage-status",
      stageCountRows.map((row) => row.label),
      stageCountRows.map((row) => row.value),
      stageCountRows.map((row) => row.color),
      { maxY: null },
    );

    const runtimeNode = document.getElementById("dashboard-runtime-overview");
    if (runtimeNode) {
      const total = stageRows.reduce((acc, row) => acc + Number(row.duration_seconds || 0), 0);
      const failed = stageRows.filter((row) => Number(row.return_code || 0) !== 0).length;
      const partialText = runtime.is_partial ? "Yes (stage artifacts still accumulating)" : "No";
      const pipelineSummary = snapshot.pipeline_summary || {};
      const outputs = pipelineSummary.outputs || {};
      const health = ((pipelineSummary.llm_runtime || {}).startup_health_check || {}).result || {};
      const healthState = health.ok === true ? "OK" : health.ok === false ? "UNAVAILABLE" : "UNKNOWN";
      const healthReason = String(health.reason || health.error_type || "").trim();
      runtimeNode.innerHTML = `
        <div class="meta-grid">
          <div><strong>Stages tracked:</strong> ${escapeHtml(String(stageRows.length))}</div>
          <div><strong>Total runtime:</strong> ${escapeHtml(total.toFixed(2))}s</div>
          <div><strong>Failed stages:</strong> ${escapeHtml(String(failed))}</div>
          <div><strong>Partial cycle:</strong> ${escapeHtml(partialText)}</div>
          <div><strong>LLM startup health:</strong> ${escapeHtml(healthState)}${healthReason ? ` (${escapeHtml(healthReason)})` : ""}</div>
          <div><strong>Initial-model runs root:</strong> <code>${escapeHtml(shortPath(outputs.initial_model_runs_root, 4))}</code></div>
        </div>
      `;
    }

    const flowNode = document.getElementById("dashboard-flow-state");
    if (flowNode) {
      const cycleStatus = String(notes.latest_cycle_status || "unknown").trim() || "unknown";
      const currentCycle = Number(session.current_cycle || 0);
      const fullStatus = String(notes.latest_full_session_status || "idle").trim() || "idle";
      const fullDone = Number(notes.latest_full_session_cycles_completed || 0);
      const fullRequested = Number(notes.latest_full_session_cycles_requested || 0);
      const awaitingFresh = Boolean(notes.awaiting_fresh_acquisition || state.awaitingFreshAcquisition);
      const requestedRefine = Boolean(notes.cycle_requested_refinement || state.cycleRequestedRefinement);
      const pipelineSummary = snapshot.pipeline_summary || {};
      const iterativeDataflow = pipelineSummary.iterative_dataflow || {};
      const iterativeMode = String(iterativeDataflow.mode || "base_pseudodata");
      const iterativeUsedPrev = Boolean(iterativeDataflow.used_previous_cycle);
      const sourceCycleRoot = shortPath(iterativeDataflow.source_cycle_root, 4);
      const initialModelRunsRoot = shortPath((pipelineSummary.outputs || {}).initial_model_runs_root, 4);
      const nextAction = !snapshot.has_model
        ? "Run model creation (Step 01 → 02)."
        : !snapshot.has_pseudodata || awaitingFresh
          ? "Acquire fresh pseudodata (Step 03) before next analysis cycle."
          : "Run analysis cycle (Step 04).";
      flowNode.innerHTML = `
        <div class="meta-grid">
          <div><strong>Current cycle:</strong> ${escapeHtml(String(currentCycle))}</div>
          <div><strong>Latest cycle status:</strong> ${escapeHtml(cycleStatus)}</div>
          <div><strong>Full-run status:</strong> ${escapeHtml(fullStatus)} (${escapeHtml(String(fullDone))}/${escapeHtml(String(fullRequested || currentCycle || 1))})</div>
          <div><strong>Refinement requested:</strong> ${requestedRefine ? "Yes" : "No"}</div>
          <div><strong>Awaiting fresh acquisition:</strong> ${awaitingFresh ? "Yes" : "No"}</div>
          <div><strong>Iterative mode:</strong> ${escapeHtml(iterativeMode)}${iterativeUsedPrev ? " (previous-cycle memory)" : " (base input)"}</div>
          <div><strong>Source cycle root:</strong> <code>${escapeHtml(sourceCycleRoot)}</code></div>
          <div><strong>Handoff model source:</strong> <code>${escapeHtml(initialModelRunsRoot)}</code></div>
        </div>
        <p class="muted compact" style="margin-top:6px"><strong>Next logical action:</strong> ${escapeHtml(nextAction)}</p>
      `;
    }

    const stageStatusNode = document.getElementById("dashboard-stage-status");
    if (stageStatusNode) {
      stageStatusNode.innerHTML = stageCountRows.length
        ? `<div class="runtime-pill-grid">${stageCountRows.map((row) => `
            <span class="runtime-pill" data-state="${escapeHtml(row.key)}">${escapeHtml(row.label)}: ${escapeHtml(String(row.value))}</span>
          `).join("")}</div>`
        : `<p class="muted compact">No stage status metrics available yet.</p>`;
    }

    const stageStatusRowsNode = document.getElementById("dashboard-stage-status-rows");
    if (stageStatusRowsNode) {
      const rows = Array.isArray(runtime.stage_status_rows) ? runtime.stage_status_rows.slice(0, 12) : [];
      if (!rows.length) {
        stageStatusRowsNode.innerHTML = `<li class="muted">No stage status rows available yet.</li>`;
      } else {
        stageStatusRowsNode.innerHTML = rows.map((row) => {
          const stageId = String(row.stage_id || "stage");
          const status = String(row.status || "unknown").toUpperCase();
          const event = String(row.event || "").trim();
          const ts = String(row.timestamp_local || "").trim();
          const msg = String(row.message || "").trim();
          const head = `${stageId} · ${status}${event ? ` · ${event}` : ""}${ts ? ` · ${ts}` : ""}`;
          return `<li><strong>${escapeHtml(head)}</strong>${msg ? ` — ${escapeHtml(msg)}` : ""}</li>`;
        }).join("");
      }
    }

    const phaseOutputsNode = document.getElementById("dashboard-phase-outputs");
    if (phaseOutputsNode) {
      const lines = [];
      const step03Count = Array.isArray(dashboard.step03?.recommended_targets) ? dashboard.step03.recommended_targets.length : 0;
      const step04Count = Array.isArray(dashboard.step04?.recommended_predictors) ? dashboard.step04.recommended_predictors.length : 0;
      const step05Targets = Array.isArray(dashboard.step05?.selected_targets) ? dashboard.step05.selected_targets.length : 0;
      const step05Barriers = Array.isArray(dashboard.step05?.selected_barriers) ? dashboard.step05.selected_barriers.length : 0;
      lines.push(`Step 03: ${dashboard.step03?.status || "unknown"} (${step03Count} targets)`);
      lines.push(`Step 04: ${dashboard.step04?.status || "unknown"} (${step04Count} updated predictors)`);
      lines.push(`Step 05: ${dashboard.step05?.status || "unknown"} (${step05Targets} targets, ${step05Barriers} barriers)`);
      const communicationStage = String(state.snapshot?.communication_summary?.payload?.stage || "none");
      lines.push(`Communication stage: ${communicationStage}`);
      phaseOutputsNode.innerHTML = lines.map((line) => `<li>${escapeHtml(line)}</li>`).join("");
    }

    const impactNode = document.getElementById("dashboard-impact-overview");
    if (impactNode) {
      const top = (impact.top_predictors || [])[0] || {};
      impactNode.innerHTML = `
        <div class="meta-grid">
          <div><strong>Status:</strong> ${escapeHtml(String(impact.status || "unknown"))}</div>
          <div><strong>Top predictor:</strong> ${escapeHtml(String(top.predictor_display || top.predictor_label || top.predictor || "—"))}</div>
        </div>
      `;
    }

    const liveEventsNode = document.getElementById("dashboard-live-events");
    if (liveEventsNode) {
      const events = Array.isArray(state.runtimeEvents) ? state.runtimeEvents.slice(-8).reverse() : [];
      if (!events.length) {
        liveEventsNode.innerHTML = `<li class="muted">No live events yet.</li>`;
      } else {
        liveEventsNode.innerHTML = events.map((item) => {
          const stamp = String(item?.time || "").trim();
          const text = String(item?.message || "").trim();
          return `<li>${escapeHtml(stamp ? `[${stamp}] ${text}` : text)}</li>`;
        }).join("");
      }
    }

    const stageEventsNode = document.getElementById("dashboard-stage-events");
    if (stageEventsNode) {
      const rows = Array.isArray(runtime.stage_event_rows) ? runtime.stage_event_rows.slice(0, 14) : [];
      if (!rows.length) {
        stageEventsNode.innerHTML = `<li class="muted">No stage artifact events written yet.</li>`;
      } else {
        stageEventsNode.innerHTML = rows.map((row) => {
          const stamp = String(row.timestamp_local || "").trim();
          const stage = String(row.stage || row.stage_id || "").trim();
          const event = String(row.event || "").trim();
          const msg = String(row.message || "").trim();
          const head = [stamp ? `[${stamp}]` : "", stage, event].filter(Boolean).join(" ");
          return `<li><strong>${escapeHtml(head)}</strong>${msg ? ` — ${escapeHtml(msg)}` : ""}</li>`;
        }).join("");
      }
    }

    const componentRuntimeNode = document.getElementById("dashboard-component-runtime");
    if (componentRuntimeNode) {
      const rows = COMPONENT_DEFS
        .filter((item) => item.id !== "cohort_batch")
        .map((item) => {
          const status = normalizeStatus(state.componentStatus[item.id] || "idle");
          const detail = String(state.componentDetail[item.id] || "").trim();
          const extra = detail && !detail.startsWith("Category:") ? ` — ${detail}` : "";
          return `<li class="detail-row-chip"><span class="status-chip status-${status}">${status.toUpperCase()}</span> ${escapeHtml(item.label)}${escapeHtml(extra)}</li>`;
        });
      componentRuntimeNode.innerHTML = rows.length
        ? rows.join("")
        : `<li class="muted">No component runtime state yet.</li>`;
    }

    const llmSignalsNode = document.getElementById("dashboard-llm-signals");
    if (llmSignalsNode) {
      const pipelineSummary = snapshot.pipeline_summary || {};
      const llmRuntime = pipelineSummary.llm_runtime || {};
      const health = (llmRuntime.startup_health_check || {}).result || {};
      const eventRows = Array.isArray(runtime.stage_event_rows) ? runtime.stage_event_rows : [];
      const liveRows = Array.isArray(state.runtimeEvents) ? state.runtimeEvents : [];
      const corpus = [
        ...eventRows.map((row) => `${row.stage || ""} ${row.event || ""} ${row.message || ""}`.toLowerCase()),
        ...liveRows.map((row) => String(row.message || "").toLowerCase()),
      ];
      const countContains = (needles) => {
        const keys = Array.isArray(needles) ? needles : [needles];
        return corpus.reduce((acc, row) => (
          keys.some((key) => row.includes(String(key).toLowerCase())) ? acc + 1 : acc
        ), 0);
      };
      const healthText = health.ok === true
        ? "OK"
        : health.ok === false
          ? `UNAVAILABLE (${health.reason || health.error_type || "unknown"})`
          : "UNKNOWN";
      const signalRows = [
        `Startup health: ${healthText}`,
        `Resolved provider: ${health.provider || "n/a"} · model=${health.resolved_model || health.requested_model || "n/a"}`,
        `Fallback mentions: ${countContains(["fallback", "deterministic", "auto-repair"])}`,
        `Critic loop mentions: ${countContains(["critic", "pass threshold", "guardrail"])}`,
        `LLM failure mentions: ${countContains(["apiconnectionerror", "provider_unavailable", "llm call failed", "authentication_failed"])}`,
      ];
      llmSignalsNode.innerHTML = signalRows.map((row) => `<li>${escapeHtml(row)}</li>`).join("");
    }

    const opNode = document.getElementById("dashboard-operationalization-overview");
    if (opNode) {
      const op = snapshot.operationalization_summary || {};
      const opVars = Array.isArray(op.variables) ? op.variables : [];
      const source = String(op.source || "none");
      const match = op.matches_session_complaint;
      const confidence = Number(op.confidence_avg_0_1 || 0);
      const domains = Array.isArray(op.ontology_domains) ? op.ontology_domains : [];
      const complaintText = String(op.complaint_text || snapshot.session?.complaint_text || "").trim();
      const preview = complaintText.length > 140 ? `${complaintText.slice(0, 140)}...` : complaintText;
      const rows = [
        `Source: ${source}`,
        `Mapped criteria: ${opVars.length} · avg confidence=${confidence.toFixed(2)}`,
        `Complaint sync: ${match === true ? "match" : match === false ? "mismatch" : "unknown"}`,
        `Domains: ${domains.length ? domains.slice(0, 4).join(", ") : "n/a"}`,
        `Input complaint: ${preview || "n/a"}`,
      ];
      if (opVars.length) {
        const top = opVars.slice(0, 3).map((row) => toDisplayLabel(row.var_id || "", { ...(labelMap || {}), [String(row.var_id || "")]: String(row.label || "") }));
        rows.push(`Top mapped signals: ${top.join(" | ")}`);
      }
      opNode.innerHTML = rows.map((row) => `<li>${escapeHtml(row)}</li>`).join("");
    }
  }

  function renderCommunication(communication) {
    const target = document.getElementById("communication-summary");
    if (!target) return;
    const payload = communication?.payload || {};
    const summary = payload.summary || {};
    const stage = String(payload.stage || "").toLowerCase();
    const showCommunication = Boolean(payload && Object.keys(payload).length && stage.startsWith("cycle_"));
    if (!showCommunication) {
      target.innerHTML = `<p class="muted">Communication agent output appears after cycle completion.</p>`;
      const previewNode = document.getElementById("control-communication-preview");
      if (previewNode) previewNode.textContent = "Communication summary will appear after cycle completion.";
      return;
    }
    const listBlock = (title, rows, emptyText) => {
      const values = Array.isArray(rows) ? rows.filter((x) => String(x || "").trim()) : [];
      return `
        <article class="detail-card">
          <h3>${escapeHtml(title)}</h3>
          ${values.length
            ? `<ul class="detail-list">${values.map((item) => `<li>${escapeHtml(String(item))}</li>`).join("")}</ul>`
            : `<p class="muted compact">${escapeHtml(emptyText)}</p>`
          }
        </article>
      `;
    };

    target.innerHTML = `
      <h3>${escapeHtml(summary.headline || "Communication Summary")}</h3>
      <p>${escapeHtml(summary.summary_markdown || "")}</p>
      <div class="meta-grid">
        <div><strong>LLM enabled:</strong> ${escapeHtml(String(payload.llm_enabled))}</div>
        <div><strong>Stage:</strong> ${escapeHtml(payload.stage || "unknown")}</div>
      </div>
      <div class="detail-grid" style="margin-top:10px">
        ${listBlock("Key Points", summary.key_points || [], "No key points reported.")}
        ${listBlock("Risks / Caveats", summary.risks || [], "No specific risks reported.")}
        ${listBlock("Recommended Next Actions", summary.recommended_next_actions || [], "No next actions reported.")}
      </div>
      <details style="margin-top:10px">
        <summary>Raw communication JSON</summary>
        <pre class="json-view">${escapeHtml(JSON.stringify(payload, null, 2))}</pre>
      </details>
    `;

    const previewNode = document.getElementById("control-communication-preview");
    if (previewNode) {
      const previewText = String(summary.headline || summary.summary_markdown || "").slice(0, 220).trim();
      previewNode.textContent = previewText || "Communication summary available.";
    }
  }

  function renderCohortSummary(cohortSummary) {
    if (!UI.cohortSummary) return;
    const payload = cohortSummary && typeof cohortSummary === "object" ? cohortSummary : {};
    if (!Object.keys(payload).length) {
      UI.cohortSummary.textContent = "No cohort run artifacts yet.";
      return;
    }
    const status = String(payload.status || "unknown").toUpperCase();
    const runId = String(payload.run_id || "n/a");
    const patientCount = Number(payload.patient_count || 0);
    const completed = Number(payload.summary?.completed || 0);
    const failed = Number(payload.summary?.failed || 0);
    UI.cohortSummary.innerHTML = `
      <div class="meta-grid">
        <div><strong>Run ID:</strong> ${escapeHtml(runId)}</div>
        <div><strong>Status:</strong> ${escapeHtml(status)}</div>
        <div><strong>Patients:</strong> ${escapeHtml(String(patientCount))}</div>
        <div><strong>Completed:</strong> ${escapeHtml(String(completed))}</div>
        <div><strong>Failed:</strong> ${escapeHtml(String(failed))}</div>
      </div>
      <p class="muted compact">Cohort artifacts are persisted per generated session under workspace sessions and cohort manifest outputs.</p>
    `;
  }

  function derivePipelineSignals() {
    const snapshot = state.snapshot || {};
    const dashboard = snapshot.pipeline_dashboard || {};
    const complaintText = String(snapshot.session?.complaint_text || "").trim();
    const hasIntake = Boolean(complaintText);
    const hasIntakeReady = hasIntake && snapshot.intake_sync?.matches === true;
    const hasModel = Boolean(snapshot.has_model);
    const hasAcquisition = Boolean(snapshot.has_pseudodata) && !state.awaitingFreshAcquisition;
    const communicationStage = String(snapshot.communication_summary?.payload?.stage || "").toLowerCase();
    const hasFinalCommunication = communicationStage.startsWith("cycle_");
    const hasAnalysis = Boolean(
      snapshot.has_pipeline_summary ||
      (dashboard.readiness && (dashboard.readiness.tier || dashboard.readiness.label !== "unknown")) ||
      (dashboard.impact?.top_predictors || []).length ||
      dashboard.step03?.status === "generated" ||
      dashboard.step04?.status === "generated"
    );
    const hasIntervention = Boolean(
      (dashboard.step05?.selected_barriers || []).length ||
      String(dashboard.step05?.user_summary || "").trim() ||
      dashboard.step05?.status === "generated" ||
      hasFinalCommunication
    );
    return { hasIntake, hasIntakeReady, hasModel, hasAcquisition, hasAnalysis, hasIntervention };
  }

  function phaseRunningFlags() {
    return {
      model: isRunningStatus(state.componentStatus.step01_operationalization) ||
        isRunningStatus(state.componentStatus.step02_initial_model),
      acquisition: isRunningStatus(state.componentStatus.pseudodata_collection) ||
        isRunningStatus(state.componentStatus.manual_data_upload),
      analysis: isRunningStatus(state.componentStatus.pipeline_cycle_engine) ||
        isRunningStatus(state.componentStatus.readiness_analysis) ||
        isRunningStatus(state.componentStatus.network_analysis) ||
        isRunningStatus(state.componentStatus.impact_quantification) ||
        isRunningStatus(state.componentStatus.step03_target_selection) ||
        isRunningStatus(state.componentStatus.step04_updated_model),
      intervention: isRunningStatus(state.componentStatus.step05_intervention) ||
        isRunningStatus(state.componentStatus.communication_agent),
    };
  }

  function phaseFailureFlags() {
    return {
      model:
        normalizeStatus(state.componentStatus.step01_operationalization) === "failed" ||
        normalizeStatus(state.componentStatus.step02_initial_model) === "failed",
      acquisition:
        normalizeStatus(state.componentStatus.pseudodata_collection) === "failed" ||
        normalizeStatus(state.componentStatus.manual_data_upload) === "failed",
      analysis:
        normalizeStatus(state.componentStatus.pipeline_cycle_engine) === "failed" ||
        normalizeStatus(state.componentStatus.readiness_analysis) === "failed" ||
        normalizeStatus(state.componentStatus.network_analysis) === "failed" ||
        normalizeStatus(state.componentStatus.impact_quantification) === "failed" ||
        normalizeStatus(state.componentStatus.step03_target_selection) === "failed" ||
        normalizeStatus(state.componentStatus.step04_updated_model) === "failed",
      intervention:
        normalizeStatus(state.componentStatus.step05_intervention) === "failed" ||
        normalizeStatus(state.componentStatus.communication_agent) === "failed",
    };
  }

  function computePhases() {
    const s = derivePipelineSignals();
    const running = phaseRunningFlags();
    const failed = phaseFailureFlags();
    const phases = [
      {
        key: "intake",
        label: "INTAKE",
        note: s.hasIntakeReady
          ? "Complaint and context synced to Step-01 input artifact"
          : "Save intake to synchronize free-text complaint artifact",
        done: s.hasIntakeReady,
        active: false,
      },
      {
        key: "model",
        label: "MODEL CREATION",
        note: "Operationalization and observation model construction",
        done: s.hasModel && !failed.model,
        active: running.model,
      },
      {
        key: "acquisition",
        label: "ACQUISITION",
        note: state.awaitingFreshAcquisition
          ? "Fresh data required for requested refinement"
          : "Collect or synthesize time-series data",
        done: s.hasAcquisition && !failed.acquisition,
        active: running.acquisition,
      },
      {
        key: "analysis",
        label: "ANALYSIS",
        note: "Readiness, network, impact, and target selection",
        done: s.hasAnalysis && !failed.analysis,
        active: running.analysis,
      },
      {
        key: "intervention",
        label: "INTERVENTION",
        note: "Step-05 intervention translation and final communication output",
        done: s.hasIntervention && !failed.intervention,
        active: running.intervention,
      },
    ];

    let activeIndex = phases.findIndex((phase) => phase.active);
    if (activeIndex === -1) {
      activeIndex = phases.findIndex((phase) => !phase.done);
    }

    const decorated = phases.map((phase, index) => {
      if (phase.key === "intake" && !s.hasIntakeReady) {
        return { ...phase, status: "active" };
      }
      const failedPhase = (phase.key === "model" && failed.model)
        || (phase.key === "acquisition" && failed.acquisition)
        || (phase.key === "analysis" && failed.analysis)
        || (phase.key === "intervention" && failed.intervention);
      if (phase.done) return { ...phase, status: "done" };
      if (failedPhase) return { ...phase, status: "error" };
      if (index === activeIndex) return { ...phase, status: "active" };
      return { ...phase, status: "pending" };
    });
    return decorated;
  }

  function executionChannelMeta(snapshot) {
    const runId = String(snapshot?.pipeline_summary?.run_id || snapshot?.session?.pipeline_run_id || "").trim();
    if (!runId) {
      return {
        label: "No cycle execution yet",
        detail: "Run from frontend controls or import CLI artifacts into this session path.",
      };
    }
    const token = runId.toLowerCase();
    const frontendRun = token.startsWith("frontend_pipeline_") || token.startsWith("frontend_");
    if (frontendRun) {
      return {
        label: "Frontend orchestrated run",
        detail: `run_id=${runId}`,
      };
    }
    return {
      label: "CLI / external orchestrated run",
      detail: `run_id=${runId}`,
    };
  }

  function flowChipClass(stateValue) {
    if (stateValue === "done") return "status-succeeded";
    if (stateValue === "running") return "status-running";
    if (stateValue === "failed") return "status-failed";
    if (stateValue === "ready") return "status-queued";
    return "status-idle";
  }

  function flowStateLabel(stateValue) {
    if (stateValue === "done") return "DONE";
    if (stateValue === "running") return "RUNNING";
    if (stateValue === "failed") return "FAILED";
    if (stateValue === "ready") return "READY";
    return "BLOCKED";
  }

  function buildFlowStages() {
    const s = derivePipelineSignals();
    const running = phaseRunningFlags();
    const failed = phaseFailureFlags();
    const snapshot = state.snapshot || {};
    const dashboard = snapshot.pipeline_dashboard || {};
    const modelSummary = snapshot.model_summary || {};
    const pseudoSummary = snapshot.pseudodata_summary || {};
    const intakeSync = state.snapshot?.intake_sync || {};
    const intakeSynced = s.hasIntakeReady;
    const complaintText = String(snapshot.session?.complaint_text || "").trim();
    const step03Count = Array.isArray(dashboard.step03?.recommended_targets) ? dashboard.step03.recommended_targets.length : 0;
    const step04Count = Array.isArray(dashboard.step04?.recommended_predictors) ? dashboard.step04.recommended_predictors.length : 0;
    const step05Targets = Array.isArray(dashboard.step05?.selected_targets) ? dashboard.step05.selected_targets.length : 0;
    const step05Barriers = Array.isArray(dashboard.step05?.selected_barriers) ? dashboard.step05.selected_barriers.length : 0;
    const readinessTier = String(dashboard.readiness?.tier || dashboard.readiness?.label || "n/a");

    const rows = [
      {
        key: "intake",
        order: 1,
        label: "Intake",
        input: "Free-text complaint + context",
        output: "free_text_complaints.txt",
        note: intakeSynced
          ? "Session complaint is synced to Step-01 input."
          : "Enter complaint text (if empty) and save intake to sync artifact.",
        done: intakeSynced,
        running: false,
        failed: false,
        ready: !intakeSynced,
        metric: complaintText ? `${complaintText.length} chars captured` : "Complaint missing; intake action required",
        actionButtonId: !intakeSynced ? "save-intake-btn" : "",
        actionKind: !intakeSynced ? "sync_intake" : "",
      },
      {
        key: "model",
        order: 2,
        label: "Model",
        input: "Synced complaint artifact",
        output: "Step-01 criteria + Step-02 initial model",
        note: "Operationalization and initial observation model construction.",
        done: s.hasModel && !failed.model,
        running: running.model,
        failed: failed.model,
        ready: !s.hasModel && intakeSynced,
        metric: s.hasModel
          ? `${Number(modelSummary.criteria_count || 0)} criteria · ${Number(modelSummary.predictor_count || 0)} predictors`
          : "Model artifacts pending",
        actionButtonId: "run-initial-model-btn",
        actionKind: "model",
      },
      {
        key: "acquisition",
        order: 3,
        label: "Data",
        input: "Initial model schema",
        output: "pseudodata_wide.csv",
        note: state.awaitingFreshAcquisition
          ? "Model refinement requested; generate fresh pseudodata first."
          : "Synthesize or upload EMA-compatible time-series data.",
        done: s.hasAcquisition && !failed.acquisition,
        running: running.acquisition,
        failed: failed.acquisition,
        ready: !s.hasAcquisition && s.hasModel,
        metric: s.hasAcquisition
          ? `${Number(pseudoSummary.n_points || 0)} points · missing ${(Number(pseudoSummary.missing_rate_target || 0) * 100).toFixed(0)}%`
          : "No usable pseudodata yet",
        actionButtonId: "synthesize-btn",
        actionKind: "acquisition",
      },
      {
        key: "analysis",
        order: 4,
        label: "Analysis",
        input: "Pseudodata + model",
        output: "Readiness/network/impact + Step-03/04",
        note: "Run integrated cycle through readiness, network, impact, and handoff.",
        done: s.hasAnalysis && !failed.analysis,
        running: running.analysis,
        failed: failed.analysis,
        ready: !s.hasAnalysis && s.hasAcquisition,
        metric: s.hasAnalysis
          ? `readiness=${readinessTier} · step03=${step03Count} · step04=${step04Count}`
          : "No HUA analysis artifacts yet",
        actionButtonId: "run-cycle-btn",
        actionKind: "analysis",
      },
      {
        key: "intervention",
        order: 5,
        label: "Intervention",
        input: "Analysis handoff outputs",
        output: "Step-05 + treatment communication",
        note: "Generate intervention package and communication summary.",
        done: s.hasIntervention && !failed.intervention,
        running: running.intervention,
        failed: failed.intervention,
        ready: !s.hasIntervention && s.hasAnalysis,
        metric: s.hasIntervention
          ? `targets=${step05Targets} · barriers=${step05Barriers}`
          : "No intervention outputs yet",
        actionButtonId: "run-cycle-btn",
        actionKind: "intervention",
      },
    ];

    rows.forEach((row) => {
      if (row.failed) {
        row.state = "failed";
      } else if (row.running) {
        row.state = "running";
      } else if (row.done) {
        row.state = "done";
      } else if (row.ready) {
        row.state = "ready";
      } else {
        row.state = "blocked";
      }
    });
    return rows;
  }

  function nextFlowAction(stages) {
    if (stages.some((row) => row.state === "running")) {
      const active = stages.find((row) => row.state === "running");
      return {
        text: `${active?.label || "Stage"} is running`,
        chip: "In progress",
        buttonId: "",
        actionKind: "",
      };
    }
    const candidate = stages.find((row) => row.state === "failed" || row.state === "ready");
    if (!candidate || !candidate.actionButtonId) {
      const blocked = stages.find((row) => row.state === "blocked");
      if (blocked) {
        return {
          text: `Waiting for ${blocked.label} prerequisites`,
          chip: `Blocked: ${blocked.label}`,
          buttonId: "",
          actionKind: "",
        };
      }
      return {
        text: "All stages complete for current cycle",
        chip: "Pipeline complete",
        buttonId: "",
        actionKind: "",
      };
    }
    const retry = candidate.state === "failed";
    return {
      text: retry ? `Retry ${candidate.label}` : `Run ${candidate.label}`,
      chip: retry ? `Retry ${candidate.label}` : `Next: ${candidate.label}`,
      buttonId: candidate.actionButtonId,
      actionKind: candidate.actionKind,
    };
  }

  function componentLabel(componentId) {
    const match = COMPONENT_DEFS.find((row) => row.id === componentId);
    return match ? match.label : String(componentId || "component");
  }

  function buildCycleLoopRows(stages) {
    const snapshot = state.snapshot || {};
    const dashboard = snapshot.pipeline_dashboard || {};
    const signals = derivePipelineSignals();
    const failures = phaseFailureFlags();
    const running = phaseRunningFlags();
    const pipelineSummary = snapshot.pipeline_summary || {};
    const iterativeDataflow = pipelineSummary.iterative_dataflow || {};
    const communicationStage = String(snapshot.communication_summary?.payload?.stage || "").toLowerCase();
    const step04CycleStatus = String(dashboard.step04?.status || "").toLowerCase();
    const step05CycleStatus = String(dashboard.step05?.status || "").toLowerCase();

    const step04Status = normalizeStatus(state.componentStatus.step04_updated_model || "");
    const step04Generated = step04CycleStatus === "generated" || step04Status === "succeeded";
    const step04Skipped = step04CycleStatus === "skipped";
    const updateModelState = failures.analysis
      ? "failed"
      : step04Status === "running"
        ? "running"
        : (step04Generated || step04Skipped)
          ? "done"
          : signals.hasAnalysis
            ? "ready"
            : "blocked";

    const dataRefreshState = failures.acquisition
      ? "failed"
      : running.acquisition
        ? "running"
        : signals.hasAcquisition
          ? "done"
          : (signals.hasModel || step04Generated)
            ? "ready"
            : "blocked";

    const huaState = failures.analysis
      ? "failed"
      : running.analysis
        ? "running"
        : signals.hasAnalysis
          ? "done"
          : signals.hasAcquisition
            ? "ready"
            : "blocked";

    const interventionState = failures.intervention
      ? "failed"
      : running.intervention
        ? "running"
        : (signals.hasIntervention || communicationStage.startsWith("cycle_") || step05CycleStatus === "skipped")
          ? "done"
          : signals.hasAnalysis
            ? "ready"
            : "blocked";

    const activeStage = stages.find((row) => row.state === "running")?.label || "none";
    const mode = String(iterativeDataflow.mode || "base_pseudodata");
    const sourceCycleRoot = shortPath(iterativeDataflow.source_cycle_root || "", 4);

    return [
      {
        label: "Updated model handoff (Step-04)",
        state: updateModelState,
        note: step04Generated
          ? "Updated predictors available for next loop."
          : step04Skipped
            ? "Skipped this cycle (no Step-03 targets)."
            : "Awaiting first cycle handoff output.",
      },
      {
        label: "Data refresh (Step-03)",
        state: dataRefreshState,
        note: state.awaitingFreshAcquisition
          ? "Fresh pseudodata required before next HUA cycle."
          : "Pseudodata artifact available for cycle execution.",
      },
      {
        label: "HUA cycle execution",
        state: huaState,
        note: `mode=${mode} · source=${sourceCycleRoot}`,
      },
      {
        label: "Intervention + communication",
        state: interventionState,
        note: step05CycleStatus === "skipped"
          ? "Skipped this cycle (handoff/intervention conditions not met)."
          : communicationStage.startsWith("cycle_")
            ? `Communication published (${communicationStage}).`
            : `Active stage: ${activeStage}`,
      },
    ];
  }

  function buildIntermediateFeedRows(stages, next) {
    const snapshot = state.snapshot || {};
    const dashboard = snapshot.pipeline_dashboard || {};
    const runtime = dashboard.runtime || {};
    const rows = [];

    const addRow = (source, message) => {
      const text = String(message || "").trim();
      if (!text) return;
      rows.push({ source, message: text });
    };

    const readiness = dashboard.readiness || {};
    const impact = dashboard.impact || {};
    const step03Targets = Array.isArray(dashboard.step03?.recommended_targets) ? dashboard.step03.recommended_targets.length : 0;
    const step04Predictors = Array.isArray(dashboard.step04?.recommended_predictors) ? dashboard.step04.recommended_predictors.length : 0;
    const step05Targets = Array.isArray(dashboard.step05?.selected_targets) ? dashboard.step05.selected_targets.length : 0;
    const step05Barriers = Array.isArray(dashboard.step05?.selected_barriers) ? dashboard.step05.selected_barriers.length : 0;
    const topPredictor = (impact.top_predictors || [])[0] || {};
    const activeComponent = Object.entries(state.componentStatus || {})
      .find(([, status]) => normalizeStatus(status) === "running");

    const doneCount = stages.filter((row) => row.state === "done").length;
    addRow("result", `Stage progress: ${doneCount}/${stages.length || 0} completed`);
    addRow("result", `Next action: ${next.text}`);
    if (activeComponent) {
      const detail = String(state.componentDetail?.[activeComponent[0]] || "").trim();
      addRow("live", `Running: ${componentLabel(activeComponent[0])}${detail ? ` — ${detail}` : ""}`);
    }
    if (snapshot.has_model) {
      addRow("result", `Model ready: criteria=${Number(snapshot.model_summary?.criteria_count || 0)} · predictors=${Number(snapshot.model_summary?.predictor_count || 0)}`);
    }
    if (snapshot.has_pseudodata) {
      addRow("result", `Data ready: points=${Number(snapshot.pseudodata_summary?.n_points || 0)} · profile=${snapshot.pseudodata_summary?.profile_id || "n/a"}`);
    }
    if (String(readiness.tier || readiness.label || "").trim()) {
      addRow("result", `Readiness: tier=${readiness.tier || readiness.label || "n/a"} · score=${Number(readiness.score_0_100 || 0).toFixed(1)}`);
    }
    if (step03Targets || step04Predictors || step05Targets || step05Barriers) {
      addRow("result", `Intermittent outputs: step03=${step03Targets} · step04=${step04Predictors} · step05 targets=${step05Targets} barriers=${step05Barriers}`);
    }
    if (topPredictor.predictor_display || topPredictor.predictor_label || topPredictor.predictor) {
      addRow("result", `Top impact predictor: ${topPredictor.predictor_display || topPredictor.predictor_label || topPredictor.predictor}`);
    }

    const stageEvents = Array.isArray(runtime.stage_event_rows) ? runtime.stage_event_rows : [];
    stageEvents.slice(0, 14).forEach((row) => {
      const stamp = String(row.timestamp_local || "").trim();
      const stage = String(row.stage || row.stage_id || "").trim();
      const event = String(row.event || "").trim();
      const msg = String(row.message || "").trim();
      const head = [stamp ? `[${stamp}]` : "", stage, event].filter(Boolean).join(" ");
      addRow("artifact", `${head}${msg ? ` — ${msg}` : ""}`);
    });

    const liveEvents = Array.isArray(state.runtimeEvents) ? state.runtimeEvents : [];
    liveEvents.slice(0, 16).forEach((item) => {
      const stamp = String(item.time || "").trim();
      const msg = String(item.message || "").trim();
      if (!msg) return;
      if (shouldSuppressLogLine(msg)) return;
      addRow("live", `${stamp ? `[${stamp}] ` : ""}${msg}`);
    });
    const tailLogs = Array.isArray(state.logLines) ? state.logLines.slice(-24) : [];
    tailLogs.slice().reverse().forEach((line) => {
      const raw = String(line || "").trim();
      if (!raw) return;
      if (shouldSuppressLogLine(raw)) return;
      const normalized = raw.length > 200 ? `${raw.slice(0, 200)}...` : raw;
      addRow("log", normalized);
    });

    const dedupe = [];
    const seen = new Set();
    rows.forEach((row) => {
      const key = `${row.source}|${row.message}`;
      if (seen.has(key)) return;
      seen.add(key);
      dedupe.push(row);
    });

    const limit = state.verbosityMode === "detailed" ? 18 : state.verbosityMode === "balanced" ? 12 : 8;
    return dedupe.slice(0, limit);
  }

  function stageComponentRatios(stages) {
    const map = {
      intake: [],
      model: ["step01_operationalization", "step02_initial_model"],
      acquisition: ["pseudodata_collection", "manual_data_upload"],
      analysis: ["pipeline_cycle_engine", "readiness_analysis", "network_analysis", "impact_quantification", "step03_target_selection", "step04_updated_model"],
      intervention: ["step05_intervention", "communication_agent"],
    };
    return stages.map((row) => {
      if (row.state === "done") return 1;
      if (row.state === "failed") return 0;
      if (row.key === "intake") return row.state === "ready" ? 0.2 : 0;
      const ids = map[row.key] || [];
      if (!ids.length) return row.state === "running" ? 0.35 : 0;
      const scores = ids.map((id) => {
        const status = normalizeStatus(state.componentStatus[id] || "idle");
        if (status === "succeeded") return 1;
        if (status === "running") return 0.45;
        if (status === "queued") return 0.2;
        return 0;
      });
      const avg = scores.reduce((acc, value) => acc + value, 0) / Math.max(1, scores.length);
      if (row.state === "running") return Math.max(avg, 0.35);
      if (row.state === "ready") return Math.max(avg, 0.18);
      return avg;
    });
  }

  function renderFlowExecutionCockpit(stages, next) {
    const total = Math.max(1, stages.length);
    const doneCount = stages.filter((row) => row.state === "done").length;
    const runningCount = stages.filter((row) => row.state === "running").length;
    const failedCount = stages.filter((row) => row.state === "failed").length;
    const readyCount = stages.filter((row) => row.state === "ready").length;
    const blockedCount = stages.filter((row) => row.state === "blocked").length;
    const ratios = stageComponentRatios(stages);
    const base = (ratios.reduce((acc, value) => acc + value, 0) / total) * 100;
    const withRunBoost = runningCount > 0 ? Math.min(99, base + 3) : base;
    const percent = failedCount === 0 && doneCount === total ? 100 : Math.max(0, Math.min(99, withRunBoost));

    if (UI.flowProgressFill) {
      UI.flowProgressFill.style.width = `${percent.toFixed(1)}%`;
      UI.flowProgressFill.classList.toggle("is-running", runningCount > 0);
      const track = UI.flowProgressFill.parentElement;
      if (track) track.setAttribute("aria-valuenow", String(Math.round(percent)));
    }
    if (UI.flowProgressLabel) {
      UI.flowProgressLabel.textContent = `${doneCount}/${total} stages complete · ${Math.round(percent)}%`;
    }
    if (UI.flowProgressDetail) {
      UI.flowProgressDetail.textContent = `running=${runningCount} · failed=${failedCount} · ready=${readyCount} · blocked=${blockedCount}`;
    }
    if (UI.flowActiveComponent) {
      const activeComponent = Object.entries(state.componentStatus || {})
        .find(([, status]) => normalizeStatus(status) === "running");
      if (activeComponent) {
        UI.flowActiveComponent.textContent = `Active component: ${componentLabel(activeComponent[0])}`;
      } else if (failedCount > 0) {
        const failedComponent = Object.entries(state.componentStatus || {})
          .find(([, status]) => normalizeStatus(status) === "failed");
        UI.flowActiveComponent.textContent = failedComponent
          ? `Active component: failure at ${componentLabel(failedComponent[0])}`
          : "Active component: failure detected";
      } else {
        UI.flowActiveComponent.textContent = `Active component: ${next.buttonId ? "ready for next stage" : "idle"}`;
      }
    }

    if (UI.flowCycleLoop) {
      const loopRows = buildCycleLoopRows(stages);
      UI.flowCycleLoop.innerHTML = loopRows.map((row) => `
        <li>
          <span class="status-chip ${flowChipClass(row.state)}">${flowStateLabel(row.state)}</span>
          <span class="loop-label">${escapeHtml(row.label)}</span>
          <span class="muted compact">${escapeHtml(row.note)}</span>
        </li>
      `).join("");
    }

    if (UI.flowIntermediateFeed) {
      const feedRows = buildIntermediateFeedRows(stages, next);
      UI.flowIntermediateFeed.innerHTML = feedRows.length
        ? feedRows.map((row) => `
            <li>
              <span class="feed-source">${escapeHtml(String(row.source || "event").toUpperCase())}</span>
              ${escapeHtml(String(row.message || ""))}
            </li>
          `).join("")
        : `<li class="muted">No intermediate results yet.</li>`;
    }
  }

  function renderFlowOrchestrator() {
    if (!UI.flowStageGrid) return;
    const snapshot = state.snapshot || {};
    const inputSync = snapshot.intake_sync || {};
    const complaint = String(snapshot.session?.complaint_text || "").trim();
    const execution = executionChannelMeta(snapshot);
    const stages = buildFlowStages();
    const next = nextFlowAction(stages);

    if (UI.flowInputChannel) {
      if (!complaint) {
        UI.flowInputChannel.textContent = "Missing complaint intake";
      } else if (inputSync.matches === true) {
        UI.flowInputChannel.textContent = "Frontend intake (synced)";
      } else {
        UI.flowInputChannel.textContent = "Frontend intake (sync required)";
      }
    }
    if (UI.flowInputDetail) {
      const path = shortPath(inputSync.complaints_path || "", 4);
      UI.flowInputDetail.textContent = inputSync.matches === true
        ? `Step-01 input artifact: ${path}`
        : `Save intake before model execution. Artifact path: ${path}`;
    }
    if (UI.flowExecutionChannel) UI.flowExecutionChannel.textContent = execution.label;
    if (UI.flowExecutionDetail) UI.flowExecutionDetail.textContent = execution.detail;

    UI.flowStageGrid.innerHTML = stages.map((row) => `
      <article class="flow-stage-card" data-state="${escapeHtml(row.state)}">
        <div class="flow-stage-head">
          <span class="flow-stage-title"><span class="flow-stage-index">${row.order}</span>${escapeHtml(row.label)}</span>
          <span class="status-chip ${flowChipClass(row.state)}">${flowStateLabel(row.state)}</span>
        </div>
        <p class="muted compact"><strong>Input:</strong> ${escapeHtml(row.input)}</p>
        <p class="muted compact"><strong>Output:</strong> ${escapeHtml(row.output)}</p>
        <p class="flow-stage-metric">${escapeHtml(String(row.metric || ""))}</p>
        <p class="flow-stage-note">${escapeHtml(row.note)}</p>
      </article>
    `).join("");

    if (UI.flowNextAction) UI.flowNextAction.textContent = next.text;
    if (UI.flowNextChip) UI.flowNextChip.textContent = next.chip;
    if (UI.flowRunNextBtn) {
      UI.flowRunNextBtn.disabled = !next.buttonId || Boolean(state.activeJobId);
      UI.flowRunNextBtn.dataset.targetButtonId = String(next.buttonId || "");
      UI.flowRunNextBtn.dataset.actionKind = String(next.actionKind || "");
      UI.flowRunNextBtn.textContent = next.buttonId ? next.text : "Run Next Step";
    }
    renderFlowExecutionCockpit(stages, next);
  }

  function renderTopbarProgress(phases) {
    var stepper = document.getElementById("pipeline-progress");
    if (!stepper) return;
    if (!phases.length) return;

    var stepNodes = stepper.querySelectorAll(".step-node");
    var connectors = stepper.querySelectorAll(".step-connector");

    phases.forEach(function (phase, i) {
      var node = stepNodes[i];
      if (!node) return;
      node.classList.remove("done", "running", "error");
      if (phase.status === "done") node.classList.add("done");
      else if (phase.status === "active") node.classList.add("running");
      else if (phase.status === "error") node.classList.add("error");
    });

    for (var i = 0; i < connectors.length; i++) {
      var conn = connectors[i];
      conn.classList.remove("done", "active");
      var left = phases[i];
      var right = phases[i + 1];
      if (left && right) {
        if (left.status === "done" && right.status === "done") conn.classList.add("done");
        else if (left.status === "done" && right.status === "active") conn.classList.add("active");
      }
    }
  }

  /* Per-component step badges: each step independently tracked */
  function updateStepStatusBadges() {
    var componentBadgeMap = [
      { componentId: "step01_operationalization", badgeId: "step01-status", panelId: "panel-step01" },
      { componentId: "step02_initial_model", badgeId: "step02-status", panelId: "panel-step02" },
      { componentId: "pseudodata_collection", badgeId: "step03-acq-status", panelId: "panel-step03" },
      { componentId: "pipeline_cycle_engine", badgeId: "step04-cycle-status", panelId: "panel-step04" },
    ];

    var snapshot = state.snapshot || {};
    /* Determine per-component state considering both live status AND snapshot */
    componentBadgeMap.forEach(function (m) {
      var liveStatus = normalizeStatus(state.componentStatus[m.componentId] || "");
      var label = "PENDING";
      var statusClass = "status-idle";
      var panelState = ""; /* empty | panel-active | panel-complete */

      /* For Step 01/02: if model exists in snapshot, they completed previously */
      if (m.componentId === "step01_operationalization" && snapshot.has_model && !liveStatus) {
        liveStatus = "succeeded";
      }
      if (m.componentId === "step02_initial_model" && snapshot.has_model && !liveStatus) {
        liveStatus = "succeeded";
      }
      if (m.componentId === "pseudodata_collection" && snapshot.has_pseudodata && !liveStatus) {
        liveStatus = "succeeded";
      }
      if (m.componentId === "pipeline_cycle_engine" && snapshot.has_pipeline_summary && !liveStatus) {
        liveStatus = "succeeded";
      }

      if (liveStatus === "succeeded" || liveStatus === "done") {
        label = "DONE"; statusClass = "status-succeeded"; panelState = "panel-complete";
      } else if (liveStatus === "running") {
        label = "RUNNING"; statusClass = "status-running"; panelState = "panel-active";
      } else if (liveStatus === "queued") {
        label = "QUEUED"; statusClass = "status-queued"; panelState = "";
      } else if (liveStatus === "failed" || liveStatus === "error") {
        label = "ERROR"; statusClass = "status-failed"; panelState = "";
      }

      /* Update badge */
      var el = document.getElementById(m.badgeId);
      if (el) {
        el.textContent = label;
        el.className = "panel-step-status status-chip " + statusClass;
      }
      /* Update panel border state */
      var panel = document.getElementById(m.panelId);
      if (panel) {
        panel.classList.remove("panel-active", "panel-complete");
        if (panelState) panel.classList.add(panelState);
      }
    });
  }

  /* Progressive disclosure: lock future steps until prerequisites are met */
  function updateStepVisibility() {
    var snapshot = state.snapshot || {};
    var hasModel = Boolean(snapshot.has_model);
    var hasPseudodata = Boolean(snapshot.has_pseudodata) && !state.awaitingFreshAcquisition;

    /* Also unlock if Step 01/02 are currently running (user should see them) */
    var step01Live = normalizeStatus(state.componentStatus.step01_operationalization || "");
    var step02Live = normalizeStatus(state.componentStatus.step02_initial_model || "");
    var modelInProgress = step01Live === "running" || step01Live === "queued" || step02Live === "running" || step02Live === "queued";

    /* Step 03: unlock when model is complete */
    var panel03 = document.getElementById("panel-step03");
    if (panel03) {
      if (hasModel || modelInProgress) {
        panel03.classList.remove("panel-locked");
      } else {
        panel03.classList.add("panel-locked");
      }
    }

    /* Step 04: unlock when data is ready */
    var panel04 = document.getElementById("panel-step04");
    if (panel04) {
      if (hasPseudodata) {
        panel04.classList.remove("panel-locked");
      } else {
        panel04.classList.add("panel-locked");
      }
    }

    /* Also render Step 01 output if criteria are available */
    renderStep01Output();
  }

  /* Render Step 01 output: prefer operationalization CSV summary, fallback to model schema */
  function renderStep01Output() {
    var outputEl = document.getElementById("step01-output");
    if (!outputEl) return;
    var snapshot = state.snapshot || {};
    var op = snapshot.operationalization_summary || {};
    var opErrors = Array.isArray(op.errors) ? op.errors.filter(function (x) { return String(x || "").trim(); }) : [];
    var source = String(op.source || "").trim();
    var criteria = Array.isArray(op.variables) ? op.variables : [];
    var hasMappedSource = source.indexOf("mapped_csv") === 0;
    if (!criteria.length) {
      if (!hasMappedSource) {
        var schema = snapshot.collection_schema || {};
        var variables = schema.variables || [];
        criteria = variables
          .filter(function (v) { return String(v.role || "").toLowerCase() === "criterion"; })
          .map(function (v) {
            return {
              var_id: String(v.var_id || ""),
              label: String(v.label || v.var_id || ""),
              ontology_path: String(v.ontology_path || ""),
              mapping_status: "MODEL_SCHEMA",
              confidence_0_1: null,
            };
          });
      }
    }

    if (!criteria.length) {
      if (hasMappedSource) {
        var complaintOnly = String(op.complaint_text || snapshot.session?.complaint_text || "").trim();
        if (complaintOnly.length > 220) complaintOnly = complaintOnly.slice(0, 220) + "...";
        var errHtml = '<div class="step-output">';
        errHtml += '<h4>⚠ Operationalization produced no mapped variables</h4>';
        errHtml += '<p class="muted compact">Source: <code>' + escapeHtml(source || "mapped_csv") + '</code></p>';
        if (complaintOnly) {
          errHtml += '<p class="muted compact"><strong>Input complaint used:</strong> ' + escapeHtml(complaintOnly) + '</p>';
        }
        if (opErrors.length) {
          errHtml += '<ul class="detail-list">';
          opErrors.slice(0, 4).forEach(function (item) {
            errHtml += '<li>' + escapeHtml(String(item)) + '</li>';
          });
          errHtml += '</ul>';
        } else {
          errHtml += '<p class="muted compact">No explicit error text was returned. Check Step-01 logs for details.</p>';
        }
        errHtml += '</div>';
        outputEl.innerHTML = errHtml;
        return;
      }
      outputEl.innerHTML = "";
      return;
    }

    var html = '<div class="step-output"><h4>✓ Mapped Criteria (' + criteria.length + ' variables)</h4>';
    var complaintPreview = String(op.complaint_text || snapshot.session?.complaint_text || "").trim();
    var sessionComplaint = String(snapshot.session?.complaint_text || "").trim();
    var mismatchWarning = String(op.mismatch_warning || "").trim();
    var hasMismatch = op.matches_session_complaint === false || Boolean(mismatchWarning);
    var avgConfidence = Number(op.confidence_avg_0_1 || 0);
    var domains = Array.isArray(op.ontology_domains) ? op.ontology_domains : [];
    if (complaintPreview.length > 220) complaintPreview = complaintPreview.slice(0, 220) + "...";
    if (sessionComplaint.length > 220) sessionComplaint = sessionComplaint.slice(0, 220) + "...";
    html += '<div class="meta-grid compact" style="margin:8px 0">';
    html += '<div><strong>Mapped:</strong> ' + criteria.length + '</div>';
    html += '<div><strong>Avg confidence:</strong> ' + avgConfidence.toFixed(2) + '</div>';
    html += '<div><strong>Ontology domains:</strong> ' + escapeHtml(domains.slice(0, 3).join(", ") || "n/a") + '</div>';
    html += '</div>';
    if (source) {
      html += '<p class="muted compact">Source: <code>' + escapeHtml(source) + '</code> · high-level ontology alignment is displayed per criterion.</p>';
    }
    if (complaintPreview) {
      html += '<p class="muted compact"><strong>Input complaint used:</strong> ' + escapeHtml(complaintPreview) + '</p>';
    }
    if (opErrors.length) {
      html += '<p class="alert error compact"><strong>Step-01 warnings:</strong> ' + escapeHtml(opErrors[0]) + '</p>';
    }
    if (hasMismatch) {
      html += '<p class="alert error compact"><strong>Warning:</strong> '
        + escapeHtml(mismatchWarning || "Mapped complaint text does not match session intake complaint.")
        + '</p>';
      if (sessionComplaint) {
        html += '<p class="muted compact"><strong>Current session complaint:</strong> ' + escapeHtml(sessionComplaint) + '</p>';
      }
    }
    html += '<ul class="criteria-list">';
    criteria.forEach(function (c) {
      var confValue = c.confidence_0_1 == null ? null : Number(c.confidence_0_1);
      var conf = confValue == null || Number.isNaN(confValue) ? "" : " · conf=" + confValue.toFixed(2);
      var status = String(c.mapping_status || "").trim();
      var meta = [String(c.ontology_path || "").trim(), status ? ("status=" + status) : "", conf].filter(Boolean).join(" | ");
      var confBar = "";
      if (confValue != null && !Number.isNaN(confValue)) {
        var pct = Math.max(0, Math.min(100, Math.round(confValue * 100)));
        confBar = '<span class="criteria-conf-bar"><span class="criteria-conf-fill" style="width:' + pct + '%"></span></span>';
      }
      html += '<li>'
        + '<span class="criteria-id">' + escapeHtml(String(c.var_id || "")) + '</span>'
        + '<span class="criteria-label">' + escapeHtml(String(c.label || c.var_id || "")) + '</span>'
        + confBar
        + '<span class="criteria-conf">' + escapeHtml(meta) + '</span>'
        + '</li>';
    });
    html += '</ul></div>';
    outputEl.innerHTML = html;
  }

  /* Update analysis sub-stage items in the decomposed Step 04 panel */
  function updateAnalysisSubStages() {
    var container = document.getElementById("analysis-sub-stages");
    if (!container) return;

    var items = container.querySelectorAll(".sub-stage-item");
    items.forEach(function (item) {
      var compId = item.getAttribute("data-component");
      if (!compId) return;
      var liveStatus = normalizeStatus(state.componentStatus[compId] || "");
      var detail = state.componentDetail[compId] || "";

      /* Determine display status */
      var label = "IDLE";
      var statusClass = "status-idle";
      if (liveStatus === "running") { label = "RUNNING"; statusClass = "status-running"; }
      else if (liveStatus === "succeeded" || liveStatus === "done") { label = "DONE"; statusClass = "status-succeeded"; }
      else if (liveStatus === "queued") { label = "QUEUED"; statusClass = "status-queued"; }
      else if (liveStatus === "failed" || liveStatus === "error") { label = "FAILED"; statusClass = "status-failed"; }

      item.setAttribute("data-status", liveStatus || "idle");
      var chip = item.querySelector(".status-chip");
      if (chip) {
        chip.textContent = label;
        chip.className = "status-chip " + statusClass;
      }
      const muted = state.verbosityMode === "concise" && (liveStatus === "idle" || liveStatus === "succeeded" || liveStatus === "done");
      item.classList.toggle("sub-stage-muted", muted);
      var detailEl = item.querySelector(".sub-stage-detail");
      /* Only overwrite detail if it's meaningful (not a category label) */
      if (detailEl && detail && !detail.startsWith("Category:")) {
        detailEl.textContent = detail;
      } else if (detailEl && state.verbosityMode === "concise" && muted) {
        detailEl.textContent = "Hidden in concise mode";
      }
    });
  }

  function renderPhaseProgress() {
    const decorated = computePhases();
    state.phaseState = decorated;
    renderTopbarProgress(decorated);
    updateStepStatusBadges();
    updateStepVisibility();
    updateAnalysisSubStages();

    const runNextLabel = UI.runNextPhaseBtn;
    if (runNextLabel) {
      const next = decorated.find((phase) => phase.status !== "done");
      if (next) {
        runNextLabel.textContent = `Run ${next.label}`;
      } else {
        runNextLabel.textContent = "Run Next Phase";
      }
    }
    renderFlowOrchestrator();
  }

  function populateCollectionSummary(collection) {
    const node = document.getElementById("collection-summary");
    if (!node) return;
    node.innerHTML = `
      <div><strong>Variables:</strong> ${collection.variable_count ?? 0}</div>
      <div><strong>Criteria:</strong> ${collection.criteria_count ?? 0}</div>
      <div><strong>Predictors:</strong> ${collection.predictor_count ?? 0}</div>
    `;
  }

  function populateCollectionTable(collection) {
    const body = document.querySelector("#collection-table tbody");
    if (!body) return;
    const rows = (collection.variables || []).map((row) => `
      <tr>
        <td><code>${escapeHtml(row.var_id)}</code><br>${escapeHtml(row.label || "")}</td>
        <td>${escapeHtml(row.role || "")}</td>
        <td>${escapeHtml(row.question || "")}</td>
        <td>${escapeHtml(row.response_scale || "")}</td>
        <td>
          <input class="baseline-input" type="number" min="0" max="1" step="0.01"
            value="${row.default_baseline_0_1 ?? 0.5}" data-var-id="${escapeHtml(row.var_id)}">
        </td>
      </tr>
    `);
    body.innerHTML = rows.join("");
  }

  function updateInspectionPanel(snapshot) {
    const setText = (id, value) => {
      const node = document.getElementById(id);
      if (node) node.textContent = String(value);
    };
    const collection = snapshot.collection_schema || {};
    setText("inspect-variables-count", collection.variable_count ?? 0);
    setText("inspect-criteria-count", collection.criteria_count ?? 0);
    setText("inspect-predictors-count", collection.predictor_count ?? 0);
    setText("inspect-visuals-count", (snapshot.visuals || []).length);

    if (!snapshot.communication_summary?.payload) {
      const previewNode = document.getElementById("control-communication-preview");
      if (previewNode) previewNode.textContent = "Communication summary appears after cycle completion.";
    }
  }

  function renderSnapshot() {
    const snapshot = state.snapshot || {};
    const session = snapshot.session || {};
    const sessionNotes = session.notes || {};
    const modelSummary = snapshot.model_summary || {};
    const collection = snapshot.collection_schema || {};
    const visuals = snapshot.visuals || [];
    const pseudoSummary = snapshot.pseudodata_summary || {};
    const pipelineSummary = snapshot.pipeline_summary || {};
    const dashboard = snapshot.pipeline_dashboard || {};
    const cohortSummary = snapshot.cohort_summary || {};
    const intakeSync = snapshot.intake_sync || {};
    state.engineFlowFromSummary = Array.isArray(pipelineSummary.engine_stage_flow) ? pipelineSummary.engine_stage_flow : [];
    state.qualityFlowFromSummary = Array.isArray(pipelineSummary.quality_and_research_flow)
      ? pipelineSummary.quality_and_research_flow
      : [];

    if (!state.activeSource) {
      state.awaitingFreshAcquisition = Boolean(sessionNotes.awaiting_fresh_acquisition);
      state.cycleRequestedRefinement = Boolean(sessionNotes.cycle_requested_refinement);
    }
    const refineToggle = document.getElementById("cycle-request-refinement");
    if (refineToggle && !state.activeSource) {
      refineToggle.checked = state.cycleRequestedRefinement;
    }

    const cycleEl = document.getElementById("current-cycle");
    if (cycleEl) cycleEl.textContent = String(session.current_cycle ?? 0);
    const intakeComplaint = document.getElementById("intake-complaint");
    const intakePerson = document.getElementById("intake-person");
    const intakeContext = document.getElementById("intake-context");
    if (document.activeElement !== intakeComplaint && intakeComplaint) {
      intakeComplaint.value = String(session.complaint_text || "");
    }
    if (document.activeElement !== intakePerson && intakePerson) {
      intakePerson.value = String(session.person_text || "");
    }
    if (document.activeElement !== intakeContext && intakeContext) {
      intakeContext.value = String(session.context_text || "");
    }
    const intakeSyncNode = document.getElementById("intake-sync-status");
    if (intakeSyncNode) {
      const hasFileComplaint = String(intakeSync.file_complaint_text || "").trim().length > 0;
      const match = intakeSync.matches === true;
      if (!hasFileComplaint) {
        intakeSyncNode.textContent = "Intake file sync: waiting for free-text artifact.";
        intakeSyncNode.className = "muted compact";
      } else if (match) {
        intakeSyncNode.textContent = "Intake file sync: OK (session complaint matches Step-01 input file).";
        intakeSyncNode.className = "alert success compact";
      } else {
        intakeSyncNode.textContent = "Intake file sync: mismatch detected. Save intake with reset before running Step 01.";
        intakeSyncNode.className = "alert error compact";
      }
    }
    const updatedEl = document.getElementById("session-updated");
    if (updatedEl) updatedEl.textContent = session.updated_at || "—";

    const modelSummaryEl = document.getElementById("model-summary");
    if (modelSummaryEl) {
      modelSummaryEl.innerHTML = snapshot.has_model
        ? `
          <div class="meta-grid">
            <div><strong>Criteria:</strong> ${modelSummary.criteria_count ?? 0}</div>
            <div><strong>Predictors:</strong> ${modelSummary.predictor_count ?? 0}</div>
          </div>
          <p>${escapeHtml(modelSummary.model_summary || "")}</p>
        `
        : `<p class="muted">No model artifacts yet. Run model creation first.</p>`;
    }

    const visualGrid = document.getElementById("visual-grid");
    if (visualGrid) {
      const note = visuals.length
        ? "Static visual artifact previews are hidden in the frontend. Use the interactive dashboard charts for analysis."
        : "Interactive dashboard charts are used instead of static image previews.";
      visualGrid.innerHTML = `<p class="muted">${escapeHtml(note)}</p>`;
    }

    populateCollectionSummary(collection);
    populateCollectionTable(collection);

    const pseudoNode = document.getElementById("pseudodata-summary");
    if (pseudoNode) {
      if (Object.keys(pseudoSummary).length) {
        var vars = Array.isArray(pseudoSummary.variables) ? pseudoSummary.variables : [];
        pseudoNode.innerHTML = `
          <div class="pseudodata-card">
            <h4>✓ Pseudodata Generated</h4>
            <div class="meta-grid">
              <div><strong>Time points:</strong> ${pseudoSummary.n_points || 0}</div>
              <div><strong>Criteria:</strong> ${pseudoSummary.criteria_count || 0}</div>
              <div><strong>Predictors:</strong> ${pseudoSummary.predictor_count || 0}</div>
              <div><strong>Missing rate:</strong> ${(Number(pseudoSummary.missing_rate_target || 0) * 100).toFixed(0)}%</div>
              <div><strong>Profile:</strong> ${escapeHtml(pseudoSummary.profile_id || "—")}</div>
            </div>
            ${vars.length ? `
              <div class="table-wrap" style="margin-top:8px">
                <table class="data-table">
                  <thead><tr><th>ID</th><th>Role</th><th>Baseline</th><th>Missing %</th><th>Scale</th></tr></thead>
                  <tbody>
                    ${vars.map(v => `
                      <tr>
                        <td><code>${escapeHtml(v.var_id)}</code></td>
                        <td>${escapeHtml(v.role)}</td>
                        <td>${Number(v.baseline_0_1 || 0).toFixed(2)}</td>
                        <td>${(Number(v.missing_rate_empirical || 0) * 100).toFixed(1)}%</td>
                        <td>${v.scale_min}–${v.scale_max}</td>
                      </tr>
                    `).join("")}
                  </tbody>
                </table>
              </div>
            ` : ""}
          </div>
        `;
      } else {
        pseudoNode.innerHTML = "";
      }
    }

    const pipeNode = document.getElementById("pipeline-summary");
    const rawWrap = document.getElementById("pipeline-summary-01_raw-wrap");
    const rawNode = document.getElementById("pipeline-summary-01_raw");
    if (pipeNode) {
      if (!Object.keys(pipelineSummary).length) {
        pipeNode.innerHTML = `<p class="muted">No cycle summary yet.</p>`;
        if (rawWrap) rawWrap.classList.add("hidden");
      } else {
        const stages = Array.isArray(pipelineSummary.stage_results) ? pipelineSummary.stage_results : [];
        const engineFlow = Array.isArray(pipelineSummary.engine_stage_flow) ? pipelineSummary.engine_stage_flow : [];
        const supportFlow = Array.isArray(pipelineSummary.quality_and_research_flow)
          ? pipelineSummary.quality_and_research_flow
          : [];
        state.engineFlowFromSummary = engineFlow;
        state.qualityFlowFromSummary = supportFlow;

        /* Filter out skipped stages — they ran during model creation, not the cycle */
        const activeEngine = engineFlow.filter(r => String(r.status || "").toLowerCase() !== "skipped");
        const activeSupport = supportFlow.filter(r => String(r.status || "").toLowerCase() !== "skipped");
        const verbosity = normalizeVerbosityMode(state.verbosityMode);

        const flowTable = (rows, title) => {
          if (!rows.length) return "";
          if (verbosity === "concise") {
            const clipped = rows.slice(0, 8);
            return `
              <h4>${escapeHtml(title)}</h4>
              <div class="flow-summary-grid">
                ${clipped.map((row) => `
                  <span class="flow-summary-pill">
                    <span class="label">${escapeHtml(String(row.label || row.stage_id || "stage"))}</span>
                    <span class="status-chip ${row.status === "succeeded" ? "status-succeeded" : row.status === "failed" ? "status-failed" : "status-idle"}">${escapeHtml(String(row.status || "unknown"))}</span>
                    <span>${escapeHtml(String(Number(row.duration_seconds || 0).toFixed(1)))}s</span>
                  </span>
                `).join("")}
              </div>
            `;
          }
          const clippedRows = verbosity === "balanced" ? rows.slice(0, 10) : rows;
          return `
            <h4>${escapeHtml(title)}</h4>
            <div class="table-wrap">
              <table class="data-table">
                <thead>
                  <tr><th>Stage</th><th>Status</th><th>Duration (s)</th></tr>
                </thead>
                <tbody>
                  ${clippedRows.map((row) => `
                    <tr>
                      <td>${escapeHtml(String(row.label || row.stage_id || "unknown"))}</td>
                      <td><span class="status-chip ${row.status === "succeeded" ? "status-succeeded" : row.status === "failed" ? "status-failed" : "status-idle"}">${escapeHtml(String(row.status || "unknown"))}</span></td>
                      <td>${escapeHtml(String(Number(row.duration_seconds || 0).toFixed(1)))}</td>
                    </tr>
                  `).join("")}
                </tbody>
              </table>
            </div>
          `;
        };
        const totalDuration = stages.reduce((s, r) => s + Number(r.duration_seconds || 0), 0);
        pipeNode.innerHTML = `
          <div class="step-output">
            <h4>✓ Cycle Complete</h4>
            <div class="meta-grid">
              <div><strong>Run ID:</strong> <code>${escapeHtml(String(pipelineSummary.run_id || "—"))}</code></div>
              <div><strong>Cycle:</strong> ${escapeHtml(String(pipelineSummary.cycle_index || "—"))}</div>
              <div><strong>Status:</strong> ${escapeHtml(String(pipelineSummary.status || "unknown"))}</div>
              <div><strong>Total duration:</strong> ${totalDuration.toFixed(1)}s</div>
            </div>
          </div>
          ${flowTable(activeEngine, "Analysis Engine Stages")}
          ${flowTable(activeSupport, "Support Stages")}
        `;
        if (rawWrap && rawNode) {
          rawNode.textContent = JSON.stringify(pipelineSummary, null, 2);
          rawWrap.classList.remove("hidden");
        }
      }
    }

    renderDashboard(dashboard);
    renderCommunication(snapshot.communication_summary || {});
    renderCohortSummary(cohortSummary);
    const communicationStage = String(snapshot.communication_summary?.payload?.stage || "").toLowerCase();
    const hasFinalCommunication = communicationStage.startsWith("cycle_");

    state.sectionVisibilityAuto = {
      step01_02: true,
      model_visuals: Boolean(snapshot.has_model),
      data_collection: Boolean(snapshot.has_model),
      cycle: Boolean(snapshot.has_model),
      cohort: true,
      dashboard: Boolean(snapshot.has_pipeline_summary || Object.keys(dashboard).length),
      communication: hasFinalCommunication,
    };
    applySectionVisibility();
    state.reconcilingSnapshot = true;
    try {
      const setIdleIfNotRunning = (componentId, detail) => {
        const current = normalizeStatus(state.componentStatus[componentId] || "idle");
        if (isRunningStatus(current)) return;
        setComponentState(componentId, "idle", detail);
      };

      if (!snapshot.has_model) {
        setIdleIfNotRunning("step01_operationalization", "Awaiting model creation");
        setIdleIfNotRunning("step02_initial_model", "Awaiting model creation");
        setIdleIfNotRunning("step02_visualization", "Awaiting model diagnostics");
      }
      if (!snapshot.has_pseudodata || state.awaitingFreshAcquisition) {
        setIdleIfNotRunning(
          "pseudodata_collection",
          state.awaitingFreshAcquisition ? "Fresh pseudodata required for next cycle" : "No pseudodata artifact yet",
        );
        setIdleIfNotRunning("manual_data_upload", "No manual upload in current cycle");
      }
      if (!snapshot.has_pipeline_summary) {
        setIdleIfNotRunning("pipeline_cycle_engine", "No cycle output yet");
        setIdleIfNotRunning("readiness_analysis", "No readiness output yet");
        setIdleIfNotRunning("network_analysis", "No network output yet");
        setIdleIfNotRunning("impact_quantification", "No impact output yet");
        setIdleIfNotRunning("step03_target_selection", "No Step-03 output yet");
        setIdleIfNotRunning("step04_updated_model", "No Step-04 output yet");
        setIdleIfNotRunning("step05_intervention", "No Step-05 output yet");
        setIdleIfNotRunning("impact_visualization", "No support visuals yet");
        setIdleIfNotRunning("evaluation_reporting", "No support reporting yet");
      }
      if (!hasFinalCommunication) {
        setIdleIfNotRunning("communication_agent", "Generated after cycle completion");
      }
      if (!Object.keys(cohortSummary).length) {
        setIdleIfNotRunning("cohort_batch", "No cohort run yet");
      }

      if (snapshot.has_model) {
        setComponentState("step01_operationalization", "succeeded", "Criteria mapped");
        setComponentState("step02_initial_model", "succeeded", "Initial model created");
        if (visuals.length > 0) {
          setComponentState("step02_visualization", "succeeded", "Visual diagnostics ready");
        }
      }

      const fullSessionStatus = String(sessionNotes.latest_full_session_status || "").toLowerCase();
      if (fullSessionStatus === "running") {
        setComponentState("full_session_pipeline", "running", "End-to-end run in progress");
      } else if (fullSessionStatus === "succeeded") {
        const doneCycles = Number(sessionNotes.latest_full_session_cycles_completed || 0);
        setComponentState("full_session_pipeline", "succeeded", `Completed (${doneCycles} cycle${doneCycles === 1 ? "" : "s"})`);
      } else if (fullSessionStatus === "failed") {
        setComponentState("full_session_pipeline", "failed", String(sessionNotes.latest_full_session_error || "Full run failed"));
      }

      if (snapshot.has_pseudodata && !state.awaitingFreshAcquisition) {
        setComponentState("pseudodata_collection", "succeeded", "Acquisition completed");
      }

      if (snapshot.has_pipeline_summary) {
        const stageRows = Array.isArray(pipelineSummary.stage_results) ? pipelineSummary.stage_results : [];
        const stageStatus = {};
        stageRows.forEach((row) => {
          const key = String(row.stage || "").trim().toLowerCase();
          if (!key) return;
          const code = Number(row.return_code);
          stageStatus[key] = Number.isFinite(code) && code === 0 ? "succeeded" : "failed";
        });
        const supportRows = Array.isArray(pipelineSummary.quality_and_research_flow)
          ? pipelineSummary.quality_and_research_flow
          : [];
        const supportStatusById = {};
        supportRows.forEach((row) => {
          const key = String(row.stage_id || "").trim();
          if (!key) return;
          supportStatusById[key] = String(row.status || "skipped").toLowerCase();
        });

        const stageOutcome = (...aliases) => {
          const cleanAliases = aliases.map((item) => String(item || "").toLowerCase()).filter(Boolean);
          for (const alias of cleanAliases) {
            if (stageStatus[alias]) return stageStatus[alias];
          }
          for (const [key, value] of Object.entries(stageStatus)) {
            if (cleanAliases.some((alias) => key.includes(alias))) return value;
          }
          return "";
        };

        const readinessOutcome = stageOutcome("readiness");
        const networkOutcome = stageOutcome("network", "time_series");
        const impactOutcome = stageOutcome("impact", "momentary");
        const handoffOutcome = stageOutcome("handoff", "target_identification");
        const interventionOutcome = stageOutcome("intervention", "digital_intervention");
        const communicationOutcome = stageOutcome("translation_communication", "treatment_translation_communication", "communication");
        const pipelineOutcome = normalizePipelineSummaryStatus(pipelineSummary.status || "");
        const coreFailed = [readinessOutcome, networkOutcome, impactOutcome, handoffOutcome, interventionOutcome, communicationOutcome]
          .some((token) => token === "failed");

        if (pipelineOutcome === "failed" || coreFailed) {
          setComponentState("pipeline_cycle_engine", "failed", "Cycle completed with failed core stages");
        } else if (pipelineOutcome === "running") {
          setComponentState("pipeline_cycle_engine", "running", "Cycle pipeline still running");
        } else {
          setComponentState("pipeline_cycle_engine", "succeeded", "Cycle completed");
        }

        if (readinessOutcome === "failed") {
          setComponentState("readiness_analysis", "failed", "Readiness stage failed");
        } else if (readinessOutcome === "succeeded" || String(dashboard.readiness?.tier || dashboard.readiness?.label || "").trim()) {
          setComponentState("readiness_analysis", "succeeded", "Readiness output available");
        } else {
          setComponentState("readiness_analysis", "idle", "Readiness stage skipped");
        }

        if (networkOutcome === "failed") {
          setComponentState("network_analysis", "failed", "Network stage failed");
        } else if (networkOutcome === "succeeded" || String(dashboard.network?.analysis_set || dashboard.network?.method_path || "").trim()) {
          setComponentState("network_analysis", "succeeded", "Network output available");
        } else {
          setComponentState("network_analysis", "idle", "Network stage skipped");
        }

        if (impactOutcome === "failed") {
          setComponentState("impact_quantification", "failed", "Impact stage failed");
        } else if ((dashboard.impact?.top_predictors || []).length > 0 || impactOutcome === "succeeded") {
          setComponentState("impact_quantification", "succeeded", "Impact output available");
        } else {
          setComponentState("impact_quantification", "idle", "Impact not generated for selected method path");
        }

        if (handoffOutcome === "failed") {
          setComponentState("step03_target_selection", "failed", "Step-03 handoff failed");
        } else if (dashboard.step03?.status === "generated") {
          setComponentState("step03_target_selection", "succeeded", "Targets selected");
        } else if (handoffOutcome === "succeeded") {
          setComponentState("step03_target_selection", "succeeded", "Handoff completed (no targets)");
        } else {
          setComponentState("step03_target_selection", "idle", "Skipped (no impact profile)");
        }

        if (handoffOutcome === "failed") {
          setComponentState("step04_updated_model", "failed", "Step-04 update failed");
        } else if (dashboard.step04?.status === "generated" && (dashboard.step04?.recommended_predictors || []).length > 0) {
          setComponentState("step04_updated_model", "succeeded", "Updated model available");
        } else if (dashboard.step04?.status === "skipped") {
          setComponentState("step04_updated_model", "idle", "Skipped (no Step-03 targets)");
        } else if (handoffOutcome === "succeeded") {
          setComponentState("step04_updated_model", "idle", "No Step-04 predictor updates emitted");
        }

        if (interventionOutcome === "failed") {
          setComponentState("step05_intervention", "failed", "Intervention stage failed");
        } else if (dashboard.step05?.status === "generated") {
          setComponentState("step05_intervention", "succeeded", "Intervention available");
        } else if (interventionOutcome === "succeeded") {
          setComponentState("step05_intervention", "idle", "Intervention completed without selected outputs");
        } else {
          setComponentState("step05_intervention", "idle", "Skipped (no handoff output)");
        }

        if (stageOutcome("visualization") === "succeeded" || supportStatusById.impact_visualization_support === "succeeded") {
          setComponentState("impact_visualization", "succeeded", "Support visuals generated");
        } else if (stageOutcome("visualization") === "failed") {
          setComponentState("impact_visualization", "failed", "Support visualization failed");
        } else {
          setComponentState("impact_visualization", "idle", "Support visualization skipped");
        }

        if (stageOutcome("reporting") === "succeeded" || supportStatusById.research_reporting_support === "succeeded") {
          setComponentState("evaluation_reporting", "succeeded", "Research report generated");
        } else if (stageOutcome("reporting") === "failed") {
          setComponentState("evaluation_reporting", "failed", "Research reporting failed");
        } else {
          setComponentState("evaluation_reporting", "idle", "Support reporting skipped");
        }

        const commStage = String(snapshot.communication_summary?.payload?.stage || "").toLowerCase();
        if (communicationOutcome === "failed") {
          setComponentState("communication_agent", "failed", "Treatment communication stage failed");
        } else if (commStage.startsWith("cycle_")) {
          setComponentState("communication_agent", "succeeded", "Summary generated");
        } else if (communicationOutcome === "succeeded") {
          setComponentState("communication_agent", "succeeded", "Communication stage completed");
        } else {
          setComponentState("communication_agent", "idle", "Generated after final cycle stage");
        }
      }

      if (Object.keys(cohortSummary).length) {
        const cohortStatus = String(cohortSummary.status || "").toLowerCase();
        if (cohortStatus === "succeeded") {
          setComponentState("cohort_batch", "succeeded", "Cohort run completed");
        } else if (cohortStatus === "failed") {
          setComponentState("cohort_batch", "failed", "Cohort run has failed patients");
        } else if (cohortStatus === "running") {
          setComponentState("cohort_batch", "running", "Cohort run in progress");
        }
      }
    } finally {
      state.reconcilingSnapshot = false;
    }
    syncRuntimeCardVisibility();

    updateInspectionPanel(snapshot);
    renderPhaseProgress();
    refreshTimeSeries(snapshot).catch((err) => {
      updateTimeSeriesStatus(`Time-series load failed: ${err.message || err}`);
    });
    syncControlStates();
  }

  function parseComponentFromLog(line) {
    const text = String(line || "");
    const heartbeatMarker = text.match(/\[component:([a-z0-9_]+)\]\s+heartbeat\s+elapsed=(\d+)s(?:\s+lines=(\d+))?(?:\s+last='([^']*)')?/i);
    if (heartbeatMarker) {
      const elapsed = String(heartbeatMarker[2] || "").trim();
      const lines = String(heartbeatMarker[3] || "").trim();
      const last = String(heartbeatMarker[4] || "").trim();
      const detailParts = [];
      if (elapsed) detailParts.push(`elapsed ${elapsed}s`);
      if (lines) detailParts.push(`lines ${lines}`);
      if (last) detailParts.push(last.slice(0, 72));
      return {
        componentId: heartbeatMarker[1],
        status: "running",
        detail: detailParts.join(" | ") || "Still running",
      };
    }
    const startMarker = text.match(/\[component:([a-z0-9_]+)\]\s+command_start\s+(.*)$/i);
    if (startMarker) {
      const detail = String(startMarker[2] || "").trim();
      return {
        componentId: startMarker[1],
        status: "running",
        detail: detail ? `Command start: ${detail.slice(0, 120)}` : "Command start",
      };
    }
    const componentMarker = text.match(/\[component:([a-z0-9_]+)\]\s+status=([a-z_]+)(?:\s+(.*))?/i);
    if (componentMarker) {
      const rawStatus = String(componentMarker[2] || "").toLowerCase();
      return {
        componentId: componentMarker[1],
        status: rawStatus === "heartbeat" ? "running" : componentMarker[2],
        detail: rawStatus === "heartbeat"
          ? `Still running ${String(componentMarker[3] || "").trim()}`
          : (componentMarker[3] || "").trim(),
      };
    }
    const low = text.toLowerCase();
    if (low.includes("llm.health_check.start")) {
      return {
        componentId: "pipeline_cycle_engine",
        status: "running",
        detail: "Running LLM startup health check",
      };
    }
    if (low.includes("llm.health_check.failed")) {
      return {
        componentId: "pipeline_cycle_engine",
        status: "running",
        detail: "LLM unavailable; fallback mode enabled where supported",
      };
    }
    if (low.includes("llm.health_check.ok") || low.includes("llm.health_check.passed")) {
      return {
        componentId: "pipeline_cycle_engine",
        status: "running",
        detail: "LLM startup health check passed",
      };
    }
    const stageEvent = low.match(
      /\b(operationalization|initial_model|pseudodata|readiness|network|impact|handoff|intervention|translation_communication|visualization|reporting|pipeline)\.(start|done|failed|skipped|dry_run|parallel_start)\b/,
    );
    if (stageEvent) {
      const stage = stageEvent[1];
      const event = stageEvent[2];
      const stageToComponent = {
        operationalization: "step01_operationalization",
        initial_model: "step02_initial_model",
        pseudodata: "pseudodata_collection",
        readiness: "readiness_analysis",
        network: "network_analysis",
        impact: "impact_quantification",
        handoff: "step03_target_selection",
        intervention: "step05_intervention",
        translation_communication: "communication_agent",
        visualization: "impact_visualization",
        reporting: "evaluation_reporting",
        pipeline: "pipeline_cycle_engine",
      };
      const componentId = stageToComponent[stage] || "pipeline_cycle_engine";
      if (event === "start" || event === "parallel_start") {
        return {
          componentId,
          status: "running",
          detail: event === "parallel_start" ? "Running in parallel branch" : "Stage started",
        };
      }
      if (event === "done") return { componentId, status: "succeeded", detail: "Stage completed" };
      if (event === "failed") return { componentId, status: "failed", detail: "Stage failed" };
      return { componentId, status: "idle", detail: "Stage skipped" };
    }
    if (low.includes("apiconnectionerror") || low.includes("provider_unavailable")) {
      return {
        componentId: state.activeJobKind === "pipeline_cycle" ? "pipeline_cycle_engine" : "step02_initial_model",
        status: "running",
        detail: "LLM provider unavailable; fallback path in use",
      };
    }
    if (low.includes("heuristic_fallback_llm_failure")) {
      return {
        componentId: "step05_intervention",
        status: "succeeded",
        detail: "Fallback intervention generated (LLM unavailable)",
      };
    }
    if (low.includes("retrying once with network_jobs=1")) {
      return {
        componentId: "network_analysis",
        status: "running",
        detail: "Parallel network run failed; retrying with single worker",
      };
    }
    if (low.includes("worker error")) {
      return {
        componentId: state.activeJobKind === "pipeline_cycle" ? "pipeline_cycle_engine" : "step02_initial_model",
        status: "failed",
        detail: "Worker error",
      };
    }
    if (low.includes("llm call start")) {
      return { componentId: "step02_initial_model", status: "running", detail: "LLM generation in progress" };
    }
    if (low.includes("llm call ok")) {
      return { componentId: "step02_initial_model", status: "running", detail: "LLM response received; validating output" };
    }
    if (low.includes("auto-repair: attempt")) {
      return { componentId: "step02_initial_model", status: "running", detail: "Auto-repair pass running" };
    }
    if (low.includes("critic loop")) {
      return { componentId: "step02_initial_model", status: "running", detail: "Critic/guardrail validation loop" };
    }
    if (low.includes("running step 01")) return { componentId: "step01_operationalization", status: "running", detail: "Processing complaint" };
    if (low.includes("running step 02")) return { componentId: "step02_initial_model", status: "running", detail: "Constructing initial model" };
    if (low.includes("generating step 02 visualizations")) return { componentId: "step02_visualization", status: "running", detail: "Rendering model figures" };
    if (low.includes("running integrated phoenix analysis cycle")) return { componentId: "pipeline_cycle_engine", status: "running", detail: "Running cycle stages" };
    if (low.includes("03_readiness_check") || low.includes("00_readiness_check")) {
      return { componentId: "readiness_analysis", status: "running", detail: "Readiness diagnostics" };
    }
    if (low.includes("04_time_series_analysis") || low.includes("01_time_series_analysis")) {
      return { componentId: "network_analysis", status: "running", detail: "Network estimation" };
    }
    if (low.includes("05_momentary_impact_coefficients") || low.includes("02_momentary_impact")) {
      return { componentId: "impact_quantification", status: "running", detail: "Impact quantification" };
    }
    if (low.includes("06_target_identification_and_model_update") || low.includes("03_treatment_target_handoff")) {
      return { componentId: "step03_target_selection", status: "running", detail: "Target handoff and model update" };
    }
    if (low.includes("step04_updated_observation_model")) return { componentId: "step04_updated_model", status: "running", detail: "Updated model synthesis" };
    if (low.includes("07_hapa_digital_intervention") || low.includes("03b_translation_digital_intervention")) {
      return { componentId: "step05_intervention", status: "running", detail: "Intervention translation" };
    }
    if (low.includes("08_treatment_translation_communication") || low.includes("06_treatment_translation_communication") || low.includes("translation_communication")) {
      return { componentId: "communication_agent", status: "running", detail: "Final treatment communication" };
    }
    if (low.includes("09_impact_visualizations") || low.includes("visualization_run_summary")) {
      return { componentId: "impact_visualization", status: "running", detail: "Generating support visuals" };
    }
    if (low.includes("10_research_reports") || low.includes("run_report.json")) {
      return { componentId: "evaluation_reporting", status: "running", detail: "Generating research report" };
    }
    if (low.includes("communication summary")) return { componentId: "communication_agent", status: "running", detail: "Communication pass" };
    return null;
  }

  function kindFromPath(path) {
    if (path.includes("/run-full")) return "full_session_pipeline";
    if (path.includes("/initial-model")) return "initial_model";
    if (path.includes("/synthesize")) return "synthesize_pseudodata";
    if (path.includes("/manual-data")) return "manual_data_upload";
    if (path.includes("/run-cycle")) return "pipeline_cycle";
    if (path.includes("/full-cohort")) return "full_cohort";
    return "";
  }

  function primeComponentsForJob(kind) {
    const list = JOB_KIND_COMPONENTS[kind] || [];
    list.forEach((componentId) => {
      const current = state.componentStatus[componentId] || "idle";
      if (!isRunningStatus(current)) setComponentState(componentId, "queued", "Queued");
    });
    if (list.length) setActiveComponent(list[0], "queued");
  }

  function finalizeComponentsForJob(kind, status) {
    const safe = normalizeStatus(status);
    const list = JOB_KIND_COMPONENTS[kind] || [];
    list.forEach((componentId) => {
      const current = state.componentStatus[componentId] || "idle";
      if (safe === "succeeded" && isRunningStatus(current)) {
        setComponentState(componentId, "succeeded", "Completed");
      }
      if (safe === "failed" && isRunningStatus(current)) {
        setComponentState(componentId, "failed", "Failed");
      }
    });
    setActiveComponent("idle", "idle");
  }

  async function refreshSnapshot() {
    const response = await apiGet(`/api/sessions/${sessionId}/snapshot`);
    state.snapshot = response.snapshot || {};
    renderSnapshot();
  }

  function clearStreamReconnectTimer() {
    if (state.streamReconnectTimer) {
      clearTimeout(state.streamReconnectTimer);
      state.streamReconnectTimer = null;
    }
  }

  async function finalizeJobFromStatus(jobId, finalStatus, kind, errorMessage = "") {
    const marker = `${jobId}:${finalStatus}`;
    if (state.lastTerminalJobMarker === marker) return;
    state.lastTerminalJobMarker = marker;
    clearStreamReconnectTimer();
    clearLiveSnapshotRefreshTimer();
    clearLiveSnapshotPollTimer();
    state.streamReconnectAttempts = 0;
    state.activeJobKind = kind || state.activeJobKind;
    setJobMeta(jobId, finalStatus);
    appendLog(`[frontend] Job ${jobId} finished with status=${finalStatus}`);
    pushRuntimeEvent(`Job ${jobId} finished with status ${finalStatus.toUpperCase()}`);

    if (errorMessage) {
      appendLog(`[frontend] Error: ${errorMessage}`);
      showError(errorMessage);
    } else {
      showError("");
    }

    if (state.activeJobKind === "pipeline_cycle" && finalStatus === "succeeded") {
      state.awaitingFreshAcquisition = state.cycleRequestedRefinement;
    }
    if (["synthesize_pseudodata", "manual_data_upload"].includes(state.activeJobKind) && finalStatus === "succeeded") {
      state.awaitingFreshAcquisition = false;
    }

    finalizeComponentsForJob(state.activeJobKind, finalStatus);

    if (state.activeSource) {
      state.activeSource.close();
      state.activeSource = null;
    }

    setButtonLoading(state.activeTriggerButtonId, false);
    state.activeTriggerButtonId = "";
    syncControlStates();
    try {
      await refreshSnapshot();
      setLiveRefreshStatus("SYNCED (FINAL)", "succeeded");
    } catch (err) {
      appendLog(`[frontend] Snapshot refresh failed: ${err.message || err}`);
      setLiveRefreshStatus("SYNC ERROR", "failed");
    }
  }

  function scheduleStreamReconnect(jobId) {
    if (state.streamReconnectTimer) return;
    state.streamReconnectAttempts += 1;
    const attempt = state.streamReconnectAttempts;
    const delayMs = Math.min(6000, 400 * (2 ** Math.min(attempt, 4)));
    state.streamReconnectTimer = window.setTimeout(async () => {
      state.streamReconnectTimer = null;
      if (state.activeJobId !== jobId) return;
      try {
        const payload = await apiGet(`/api/jobs/${jobId}`);
        const job = payload.job || {};
        const status = normalizeStatus(job.status || "running");
        if (status === "succeeded" || status === "failed") {
          await finalizeJobFromStatus(jobId, status, String(job.kind || state.activeJobKind), String(job.error || ""));
          return;
        }
        const cursor = Number(state.streamCursorByJob[jobId] || 0);
        appendLog(`[frontend] Reconnecting stream for ${jobId} from cursor=${cursor} (attempt ${attempt}).`);
        attachStream(jobId, String(job.kind || state.activeJobKind), cursor, true);
      } catch (err) {
        appendLog(`[frontend] Stream reconnect check failed: ${err.message || err}`);
        scheduleStreamReconnect(jobId);
      }
    }, delayMs);
  }

  function attachStream(jobId, kindHint = "", afterCursor = 0, reconnecting = false) {
    if (state.activeSource) {
      state.activeSource.close();
      state.activeSource = null;
    }
    clearStreamReconnectTimer();
    clearLiveSnapshotRefreshTimer();
    state.activeJobId = jobId;
    state.activeJobKind = kindHint || state.activeJobKind;
    const seedCursor = Math.max(0, Number(afterCursor || state.streamCursorByJob[jobId] || 0) || 0);
    state.streamCursorByJob[jobId] = seedCursor;
    setJobMeta(jobId, "running");
    syncControlStates();
    if (reconnecting) {
      appendLog(`[frontend] Reconnected to job ${jobId}.`);
      pushRuntimeEvent(`Reconnected to job ${jobId}`);
    } else {
      appendLog(`[frontend] Connected to job ${jobId}`);
      pushRuntimeEvent(`Connected to job ${jobId}`);
      if (typeof switchTab === 'function') switchTab('logs');
    }
    startLiveSnapshotPolling();
    scheduleSnapshotRefresh("stream", true);

    const streamUrl = seedCursor > 0
      ? `/api/jobs/${jobId}/stream?after=${seedCursor}`
      : `/api/jobs/${jobId}/stream`;
    state.activeSource = new EventSource(streamUrl);
    state.activeSource.onmessage = (event) => {
      let payload = {};
      try {
        payload = JSON.parse(event.data || "{}");
      } catch {
        payload = {};
      }

      if (payload.type === "log") {
        const entry = payload.entry || {};
        const idx = Number(entry.index);
        if (Number.isFinite(idx)) {
          state.streamCursorByJob[jobId] = Math.max(Number(state.streamCursorByJob[jobId] || 0), idx);
        }
        const line = String(entry.line || "");
        const low = line.toLowerCase();
        appendLog(`[${entry.timestamp || ""}] [${entry.level || "INFO"}] ${line}`);
        const parsed = parseComponentFromLog(line);
        if (parsed?.componentId) {
          setComponentState(parsed.componentId, parsed.status || "running", parsed.detail || line.slice(0, 140));
          setActiveComponent(parsed.componentId, parsed.status || "running");
          const parsedStatus = normalizeStatus(parsed.status || "running");
          if (parsedStatus !== "idle") {
            renderPhaseProgress();
          }
          if (parsedStatus !== "running" && parsedStatus !== "queued") {
            scheduleSnapshotRefresh(`${parsed.componentId}.${parsedStatus}`, false);
          }
        }
        if (low.includes("handoff.start")) {
          setComponentState("step04_updated_model", "running", "Updated model synthesis");
        }
        if (low.includes("handoff.done")) {
          setComponentState("step04_updated_model", "succeeded", "Updated model available");
          scheduleSnapshotRefresh("handoff.done", false);
        }
        if (low.includes("cohort_batch")) {
          setComponentState("cohort_batch", parsed?.status || "running", parsed?.detail || "Cohort orchestration");
          setActiveComponent("cohort_batch", parsed?.status || "running");
        }
        return;
      }

      if (payload.type === "heartbeat") {
        const cursor = Number(payload.cursor);
        if (Number.isFinite(cursor)) {
          state.streamCursorByJob[jobId] = Math.max(Number(state.streamCursorByJob[jobId] || 0), cursor);
        }
        return;
      }

      if (payload.type === "status") {
        const finalStatus = normalizeStatus(payload.status || "idle");
        const cursor = Number(payload.cursor);
        if (Number.isFinite(cursor)) {
          state.streamCursorByJob[jobId] = Math.max(Number(state.streamCursorByJob[jobId] || 0), cursor);
        }
        finalizeJobFromStatus(jobId, finalStatus, String(payload.kind || state.activeJobKind), String(payload.error || ""))
          .catch((err) => appendLog(`[frontend] Finalize job failed: ${err.message || err}`));
      }
    };

    state.activeSource.onerror = () => {
      if (state.activeJobId !== jobId) return;
      if (state.activeSource) {
        state.activeSource.close();
        state.activeSource = null;
      }
      appendLog(`[frontend] Stream disconnected for ${jobId}; retrying…`);
      scheduleStreamReconnect(jobId);
    };
  }

  async function launchJob(path, payload, triggerButtonId, processingLabel) {
    const kind = kindFromPath(path);
    const summarizeLaunchPayload = (jobKind, body) => {
      const p = body || {};
      if (jobKind === "initial_model") {
        return `model=${p.llm_model || "gpt-5-nano"} | workers=${p.max_workers || 12} | step01_leaf_adjudication=${Boolean(p.operationalization_enable_llm_rerank)} | step01_critic=${p.operationalization_critic_max_iterations || 2}@${p.operationalization_critic_pass_threshold ?? 0.78} | disable_llm=${Boolean(p.disable_llm)} | hard_ontology=${Boolean(p.hard_ontology_constraint)}`;
      }
      if (jobKind === "full_session_pipeline") {
        return `cycles=${p.cycles || 1} | model=${p.llm_model || "gpt-5-nano"} | step01_leaf_adjudication=${Boolean(p.operationalization_enable_llm_rerank)} | step01_critic=${p.operationalization_critic_max_iterations || 2}@${p.operationalization_critic_pass_threshold ?? 0.78} | net_jobs=${p.network_jobs || 1} | net_policy=${p.network_execution_policy || "readiness_aligned"} | disable_llm=${Boolean(p.disable_llm)}`;
      }
      if (jobKind === "pipeline_cycle") {
        return `model=${p.llm_model || "gpt-5-nano"} | net_jobs=${p.network_jobs || 1} | net_policy=${p.network_execution_policy || "readiness_aligned"} | parallel=${Boolean(p.parallel_branches)} | intervention=${Boolean(p.include_intervention)} | communication=${Boolean(p.run_treatment_communication)} | disable_llm=${Boolean(p.disable_llm)}`;
      }
      if (jobKind === "synthesize_pseudodata") {
        return `points=${p.n_points || 84} | missing_rate=${p.missing_rate ?? 0.1} | seed=${p.seed ?? 42}`;
      }
      if (jobKind === "full_cohort") {
        return `patients=${p.patient_count || 10} | parallel_patients=${p.parallel_patients || 2} | step01_leaf_adjudication=${Boolean(p.operationalization_enable_llm_rerank)} | step01_critic=${p.operationalization_critic_max_iterations || 2}@${p.operationalization_critic_pass_threshold ?? 0.78} | net_policy=${p.network_execution_policy || "readiness_aligned"} | disable_llm=${Boolean(p.disable_llm)}`;
      }
      return "";
    };
    appendLog(`[frontend] Launching job at ${path}`);
    const payloadSummary = summarizeLaunchPayload(kind, payload);
    if (payloadSummary) {
      appendLog(`[frontend] Launch config: ${payloadSummary}`);
    }
    pushRuntimeEvent(`Launching ${kind || "job"}`);
    setLiveRefreshStatus("LIVE SYNC PENDING", "queued");
    if (payloadSummary) {
      pushRuntimeEvent(payloadSummary);
    }
    showError("");
    if (kind) primeComponentsForJob(kind);
    state.activeTriggerButtonId = triggerButtonId || "";
    setButtonLoading(triggerButtonId, true, processingLabel || "Processing…");
    if (kind === "pipeline_cycle") {
      state.cycleRequestedRefinement = Boolean(payload?.request_model_refinement);
    }
    const result = await apiPost(path, payload);
    state.lastTerminalJobMarker = "";
    state.streamCursorByJob[result.job_id] = 0;
    attachStream(result.job_id, kind);
  }

  function bindActions() {
    document.getElementById("run-full-btn")?.addEventListener("click", async () => {
      try {
        await launchJob(
          `/api/sessions/${sessionId}/jobs/run-full`,
          {
            llm_model: document.getElementById("full-llm-model")?.value || "gpt-5-nano",
            disable_llm: Boolean(document.getElementById("full-disable-llm")?.checked),
            hard_ontology_constraint: Boolean(document.getElementById("full-hard-ontology")?.checked),
            prompt_budget_tokens: Number(document.getElementById("full-prompt-budget")?.value || 400000),
            critic_max_iterations: Number(document.getElementById("full-critic-iterations")?.value || 2),
            critic_pass_threshold: Number(document.getElementById("full-critic-threshold")?.value || 0.74),
            max_workers: Number(document.getElementById("full-max-workers")?.value || 12),
            operationalization_enable_llm_rerank: Boolean(document.getElementById("full-operationalization-rerank")?.checked),
            operationalization_critic_max_iterations: Number(document.getElementById("full-operationalization-critic-iterations")?.value || 2),
            operationalization_critic_pass_threshold: Number(document.getElementById("full-operationalization-critic-threshold")?.value || 0.78),
            n_points: Number(document.getElementById("full-points")?.value || 84),
            missing_rate: Number(document.getElementById("full-missing")?.value || 0.1),
            seed: Number(document.getElementById("full-seed")?.value || 42),
            cycles: Number(document.getElementById("full-cycles")?.value || 1),
            include_intervention: Boolean(document.getElementById("full-include-intervention")?.checked),
            request_model_refinement: Boolean(document.getElementById("full-request-refinement")?.checked),
            auto_refresh_pseudodata_each_cycle: Boolean(document.getElementById("full-auto-refresh-acquisition")?.checked),
            profile_memory_window: Number(document.getElementById("full-memory-window")?.value || 3),
            handoff_critic_max_iterations: Number(document.getElementById("cycle-handoff-critic-iterations")?.value || 2),
            handoff_critic_pass_threshold: Number(document.getElementById("cycle-handoff-critic-threshold")?.value || 0.74),
            intervention_critic_max_iterations: Number(document.getElementById("cycle-intervention-critic-iterations")?.value || 2),
            intervention_critic_pass_threshold: Number(document.getElementById("cycle-intervention-critic-threshold")?.value || 0.74),
            network_boot: Number(document.getElementById("full-network-boot")?.value || 40),
            network_block_len: Number(document.getElementById("full-network-block-len")?.value || 14),
            network_jobs: Number(document.getElementById("full-network-jobs")?.value || 4),
            network_execution_policy: document.getElementById("full-network-policy")?.value || "readiness_aligned",
            run_impact_visualizations: Boolean(document.getElementById("full-run-visualizations")?.checked),
            run_treatment_communication: Boolean(document.getElementById("full-run-communication")?.checked),
            parallel_branches: Boolean(document.getElementById("full-parallel-branches")?.checked),
          },
          "run-full-btn",
          "Running full pipeline…",
        );
      } catch (err) {
        appendLog(`[frontend] ${err.message}`);
        setButtonLoading("run-full-btn", false);
        state.activeTriggerButtonId = "";
        setJobMeta(state.activeJobId, "failed");
        showError(err.message);
        syncControlStates();
      }
    });

    document.getElementById("run-initial-model-btn")?.addEventListener("click", async () => {
      try {
        await launchJob(
          `/api/sessions/${sessionId}/jobs/initial-model`,
          {
            llm_model: document.getElementById("initial-llm-model")?.value || "gpt-5-nano",
            disable_llm: Boolean(document.getElementById("initial-disable-llm")?.checked),
            prompt_budget_tokens: Number(document.getElementById("initial-prompt-budget")?.value || 400000),
            max_workers: Number(document.getElementById("initial-max-workers")?.value || 12),
            critic_max_iterations: Number(document.getElementById("initial-critic-iterations")?.value || 2),
            critic_pass_threshold: Number(document.getElementById("initial-critic-threshold")?.value || 0.74),
            operationalization_enable_llm_rerank: Boolean(document.getElementById("initial-operationalization-rerank")?.checked),
            operationalization_critic_max_iterations: Number(document.getElementById("initial-operationalization-critic-iterations")?.value || 2),
            operationalization_critic_pass_threshold: Number(document.getElementById("initial-operationalization-critic-threshold")?.value || 0.78),
            hard_ontology_constraint: Boolean(document.getElementById("initial-hard-ontology")?.checked),
          },
          "run-initial-model-btn",
          "Processing model creation…",
        );
      } catch (err) {
        appendLog(`[frontend] ${err.message}`);
        setButtonLoading("run-initial-model-btn", false);
        state.activeTriggerButtonId = "";
        setJobMeta(state.activeJobId, "failed");
        showError(err.message);
        syncControlStates();
      }
    });

    document.getElementById("synthesize-btn")?.addEventListener("click", async () => {
      try {
        await launchJob(
          `/api/sessions/${sessionId}/jobs/synthesize`,
          {
            n_points: Number(document.getElementById("synth-points")?.value || 84),
            missing_rate: Number(document.getElementById("synth-missing")?.value || 0.1),
            seed: Number(document.getElementById("synth-seed")?.value || 42),
            baselines: readBaselines(),
          },
          "synthesize-btn",
          "Synthesizing data…",
        );
      } catch (err) {
        appendLog(`[frontend] ${err.message}`);
        setButtonLoading("synthesize-btn", false);
        state.activeTriggerButtonId = "";
        setJobMeta(state.activeJobId, "failed");
        showError(err.message);
        syncControlStates();
      }
    });

    document.getElementById("manual-upload-btn")?.addEventListener("click", async () => {
      const csvText = document.getElementById("manual-csv")?.value || "";
      if (!csvText.trim()) {
        appendLog("[frontend] Manual CSV is empty.");
        return;
      }
      try {
        await launchJob(
          `/api/sessions/${sessionId}/jobs/manual-data`,
          { csv_text: csvText },
          "manual-upload-btn",
          "Saving data…",
        );
      } catch (err) {
        appendLog(`[frontend] ${err.message}`);
        setButtonLoading("manual-upload-btn", false);
        state.activeTriggerButtonId = "";
        setJobMeta(state.activeJobId, "failed");
        showError(err.message);
        syncControlStates();
      }
    });

    document.getElementById("run-cycle-btn")?.addEventListener("click", async () => {
      const requestRefinement = Boolean(document.getElementById("cycle-request-refinement")?.checked);
      const includeIntervention = Boolean(document.getElementById("cycle-include-intervention")?.checked);
      if (state.awaitingFreshAcquisition) {
        const msg = "Fresh data collection is required before running the next HUA cycle because the model was updated.";
        appendLog(`[frontend] ${msg}`);
        pushRuntimeEvent("Cycle blocked: acquire fresh pseudodata before next HUA run");
        showError(msg);
        return;
      }
      if (!Boolean(state.snapshot?.has_pseudodata)) {
        const msg = "No pseudodata found. Run Data Collection (Step 03) before the HUA cycle.";
        appendLog(`[frontend] ${msg}`);
        showError(msg);
        return;
      }
      try {
        await launchJob(
          `/api/sessions/${sessionId}/jobs/run-cycle`,
          {
            llm_model: document.getElementById("cycle-llm-model")?.value || "gpt-5-nano",
            profile_memory_window: Number(document.getElementById("cycle-memory-window")?.value || 3),
            handoff_critic_max_iterations: Number(document.getElementById("cycle-handoff-critic-iterations")?.value || 2),
            handoff_critic_pass_threshold: Number(document.getElementById("cycle-handoff-critic-threshold")?.value || 0.74),
            intervention_critic_max_iterations: Number(document.getElementById("cycle-intervention-critic-iterations")?.value || 2),
            intervention_critic_pass_threshold: Number(document.getElementById("cycle-intervention-critic-threshold")?.value || 0.74),
            network_boot: Number(document.getElementById("cycle-network-boot")?.value || 40),
            network_block_len: Number(document.getElementById("cycle-network-block-len")?.value || 14),
            network_jobs: Number(document.getElementById("cycle-network-jobs")?.value || 4),
            network_execution_policy: document.getElementById("cycle-network-policy")?.value || "readiness_aligned",
            run_impact_visualizations: Boolean(document.getElementById("cycle-run-visualizations")?.checked),
            run_treatment_communication: Boolean(document.getElementById("cycle-run-communication")?.checked),
            parallel_branches: Boolean(document.getElementById("cycle-parallel-branches")?.checked),
            hard_ontology_constraint: Boolean(document.getElementById("cycle-hard-ontology")?.checked),
            disable_llm: Boolean(document.getElementById("cycle-disable-llm")?.checked),
            request_model_refinement: requestRefinement,
            include_intervention: includeIntervention,
          },
          "run-cycle-btn",
          requestRefinement ? "Running analysis + refinement…" : "Running analysis…",
        );
      } catch (err) {
        appendLog(`[frontend] ${err.message}`);
        setButtonLoading("run-cycle-btn", false);
        state.activeTriggerButtonId = "";
        setJobMeta(state.activeJobId, "failed");
        showError(err.message);
        syncControlStates();
      }
    });

    document.getElementById("run-cohort-btn")?.addEventListener("click", async () => {
      try {
        await launchJob(
          `/api/sessions/${sessionId}/jobs/full-cohort`,
          {
            patient_count: Number(document.getElementById("cohort-patient-count")?.value || 10),
            parallel_patients: Number(document.getElementById("cohort-parallel-patients")?.value || 2),
            llm_model: document.getElementById("cohort-llm-model")?.value || "gpt-5-nano",
            disable_llm: Boolean(document.getElementById("cohort-disable-llm")?.checked),
            operationalization_enable_llm_rerank: Boolean(document.getElementById("cohort-operationalization-rerank")?.checked),
            operationalization_critic_max_iterations: Number(document.getElementById("cohort-operationalization-critic-iterations")?.value || 2),
            operationalization_critic_pass_threshold: Number(document.getElementById("cohort-operationalization-critic-threshold")?.value || 0.78),
            n_points: Number(document.getElementById("cohort-pseudodata-points")?.value || 84),
            missing_rate: Number(document.getElementById("cohort-pseudodata-missing")?.value || 0.1),
            seed: Number(document.getElementById("cohort-pseudodata-seed")?.value || 42),
            include_intervention: Boolean(document.getElementById("cohort-include-intervention")?.checked),
            run_impact_visualizations: Boolean(document.getElementById("cohort-run-visualizations")?.checked),
            run_treatment_communication: Boolean(document.getElementById("cohort-run-communication")?.checked),
            parallel_branches: Boolean(document.getElementById("cohort-parallel-branches")?.checked),
          },
          "run-cohort-btn",
          "Running cohort pipeline…",
        );
      } catch (err) {
        appendLog(`[frontend] ${err.message}`);
        setButtonLoading("run-cohort-btn", false);
        state.activeTriggerButtonId = "";
        setJobMeta(state.activeJobId, "failed");
        showError(err.message);
        syncControlStates();
      }
    });

    document.getElementById("save-intake-btn")?.addEventListener("click", async () => {
      const complaint = String(document.getElementById("intake-complaint")?.value || "").trim();
      if (!complaint) {
        appendLog("[frontend] Complaint text is required.");
        showError("Complaint text is required.");
        return;
      }
      const person = String(document.getElementById("intake-person")?.value || "");
      const context = String(document.getElementById("intake-context")?.value || "");
      const resetOutputs = Boolean(document.getElementById("intake-reset-outputs")?.checked);
      try {
        setButtonLoading("save-intake-btn", true, "Saving intake…");
        showError("");
        const response = await fetch(`/api/sessions/${sessionId}/intake`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            complaint_text: complaint,
            person_text: person,
            context_text: context,
            reset_outputs: resetOutputs,
          }),
        });
        const payload = await response.json();
        if (!response.ok || payload.status === "error") {
          throw new Error(payload.message || `HTTP ${response.status}`);
        }
        appendLog(`[frontend] Intake updated (reset_outputs=${resetOutputs}).`);
        pushRuntimeEvent("Session intake updated");
        state.activeJobId = null;
        state.activeJobKind = "";
        setJobMeta("none", "idle");
        await refreshSnapshot();
      } catch (err) {
        appendLog(`[frontend] ${err.message || err}`);
        showError(err.message || String(err));
      } finally {
        setButtonLoading("save-intake-btn", false);
        syncControlStates();
      }
    });

    document.getElementById("timeseries-variable-select")?.addEventListener("change", () => {
      renderTimeSeriesChart();
    });
    document.getElementById("timeseries-window-select")?.addEventListener("change", () => {
      renderTimeSeriesChart();
    });
    document.getElementById("timeseries-normalize")?.addEventListener("change", () => {
      renderTimeSeriesChart();
    });

    const triggerNextPhase = () => {
      const phases = state.phaseState || [];
      const next = phases.find((phase) => phase.status !== "done");
      if (!next) {
        appendLog("[frontend] All phases completed for the current iteration.");
        return;
      }
      if (next.key === "intake") {
        document.getElementById("save-intake-btn")?.click();
        return;
      }
      if (next.key === "model") {
        document.getElementById("run-initial-model-btn")?.click();
        return;
      }
      if (next.key === "acquisition") {
        document.getElementById("synthesize-btn")?.click();
        return;
      }
      if (next.key === "analysis" || next.key === "intervention") {
        const includeInterventionToggle = document.getElementById("cycle-include-intervention");
        if (includeInterventionToggle && next.key === "intervention") {
          includeInterventionToggle.checked = true;
        }
        document.getElementById("run-cycle-btn")?.click();
        return;
      }
      appendLog(`[frontend] ${next.label} has no executable action.`);
    };

    UI.runNextPhaseBtn?.addEventListener("click", triggerNextPhase);
    UI.flowRunNextBtn?.addEventListener("click", () => {
      const actionKind = String(UI.flowRunNextBtn?.dataset.actionKind || "");
      const targetButtonId = String(UI.flowRunNextBtn?.dataset.targetButtonId || "");
      if (actionKind === "intervention") {
        const includeInterventionToggle = document.getElementById("cycle-include-intervention");
        if (includeInterventionToggle) includeInterventionToggle.checked = true;
      }
      if (targetButtonId) {
        document.getElementById(targetButtonId)?.click();
        return;
      }
      triggerNextPhase();
    });
    UI.flowOpenLogsBtn?.addEventListener("click", () => {
      if (typeof switchTab === "function") switchTab("logs");
    });
    UI.verbosityModeSelect?.addEventListener("change", () => {
      const value = String(UI.verbosityModeSelect?.value || "concise");
      setVerbosityMode(value, true);
      renderSnapshot();
    });

    /* Drawer button bindings removed — replaced by tab-based layout */
  }

  initRuntimeMap();
  initCollapsibleSections();
  initSectionVisibilityControls();
  try {
    setVerbosityMode(localStorage.getItem(VERBOSITY_STORAGE_KEY) || "concise", false);
  } catch {
    setVerbosityMode("concise", false);
  }
  try {
    if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      state.networkAnimPlaying = false;
    }
  } catch {
    // Ignore media-query lookup failures.
  }
  bindActions();
  bindModelCatalogAutocomplete();
  setJobMeta("none", "idle");
  setLiveRefreshStatus("LIVE SYNC IDLE", "idle");
  setActiveComponent("idle", "idle");
  renderRuntimeEvents();
  renderSnapshot();
})();

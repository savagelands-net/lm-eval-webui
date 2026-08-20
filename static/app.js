const state = {
	models: [],
	tasks: [],
	jobs: [],
	jobSuiteFilter: "all",
	rows: [],
	leaderboard: [],
	leaderboardSort: {},
	benchmarkProfiles: [],
	resultProfile: "all",
	selectedJobs: new Set(),
	expandedJobs: new Set(),
	jobDetails: new Map(),
	jobCommands: new Map(),
	jobLogs: new Map(),
	jobLogElements: new Map(),
	loadedResultSuites: new Set(),
	jobsLoaded: false,
	selectedModels: new Set(),
	selectedTasks: new Set(),
	visibleTaskNames: [],
	hasAutoSelectedTask: false,
	taskPage: 0,
	taskLoadToken: 0,
	activeSuite: "lemonade_bench",
	resultSuite: "lemonade_bench",
	detailFilters: new Map(),
};

const $ = (id) => document.getElementById(id);
const TASKS_PER_PAGE = 250;
const SVG_NS = "http://www.w3.org/2000/svg";
const LEADERBOARD_CATEGORIES = [
	"Reasoning",
	"Math",
	"Coding / Structured Output",
	"Instruction Following",
	"Other",
];
const PROFILE_OPTION_CONTROLS = {
	limit: "limit",
	num_fewshot: "numFewshot",
	batch_size: "batchSize",
	max_gen_toks: "maxGenToks",
	num_concurrent: "numConcurrent",
	timeout: "timeout",
	apply_chat_template: "applyChatTemplate",
	fewshot_as_multiturn: "fewshotAsMultiturn",
	log_samples: "logSamples",
	task_batch_size: "taskBatchSize",
	max_concurrent_jobs: "maxConcurrentJobs",
};
const PROFILE_BOOLEAN_OPTIONS = new Set([
	"apply_chat_template",
	"fewshot_as_multiturn",
	"log_samples",
]);
const PROFILE_NUMERIC_OPTIONS = new Set([
	"limit",
	"num_fewshot",
	"max_gen_toks",
	"num_concurrent",
	"timeout",
	"task_batch_size",
	"max_concurrent_jobs",
]);
const TASK_CATEGORY_FILTERS = [
	{ id: "taskCategoryReasoning", category: "Reasoning" },
	{ id: "taskCategoryMath", category: "Math" },
	{ id: "taskCategoryCoding", category: "Coding / Structured Output" },
	{ id: "taskCategoryInstruction", category: "Instruction Following" },
	{ id: "taskCategoryOther", category: "Other" },
];
const CLIENT_BACKENDS = new Set([
	"openai-compatible-chat-completions",
	"lemonade-chat-completions",
]);
const SUITES = {
	lemonade_bench: "Lemonade Bench",
	lm_eval: "lm-eval",
	swe_mini: "SWE Mini",
};
const DETAIL_FILTER_CONFIG = {
	models: {
		label: "Models",
		allLabel: "All models",
		optionsId: "detailModelOptions",
		summaryId: "detailModelSummary",
	},
	tasks: {
		label: "Tasks",
		allLabel: "All tasks",
		optionsId: "detailTaskOptions",
		summaryId: "detailTaskSummary",
	},
	metrics: {
		label: "Metrics",
		allLabel: "All metrics",
		optionsId: "detailMetricOptions",
		summaryId: "detailMetricSummary",
	},
};
const DEFAULT_LEMONADE_BENCH_RUNS = 3;
const DEFAULT_LEMONADE_BENCH_TIMEOUT = 300;
const DEFAULT_SWE_JUDGE_MODEL = "gpt-oss-120b-mxfp-GGUF";
const DEFAULT_SWE_TIMEOUT_MINUTES = 60;
const DEFAULT_SWE_CONTEXT_WINDOW = 65536;
const DEFAULT_SWE_MAX_OUTPUT_TOKENS = 16384;
const DEFAULT_SWE_PROVIDER_TIMEOUT_MINUTES = 15;
const ACTIVE_JOB_STATUSES = new Set(["queued", "running", "cancelling"]);
const TERMINAL_JOB_STATUSES = new Set(["cancelled", "failed", "succeeded"]);
const RESULT_PAGE_SIZE = 2000;
const JOB_POLL_INTERVAL_MS = 5000;
const REQUEST_TIMEOUT_MS = 30000;
const TASK_REQUEST_TIMEOUT_MS = 120000;
const inFlightRequests = new Map();
let jobPollTimer = null;
let jobPollFailures = 0;

async function api(path, options = {}) {
	const { timeoutMs = REQUEST_TIMEOUT_MS, ...fetchOptions } = options;
	const controller = new AbortController();
	const timeout = setTimeout(() => controller.abort(), timeoutMs);
	try {
		const response = await fetch(path, {
			headers: {
				"Content-Type": "application/json",
				...(fetchOptions.headers || {}),
			},
			...fetchOptions,
			signal: fetchOptions.signal || controller.signal,
		});
		const text = await response.text();
		let payload = {};
		if (text) {
			try {
				payload = JSON.parse(text);
			} catch (_error) {
				throw new Error(`Invalid response from ${path}`);
			}
		}
		if (!response.ok) throw new Error(payload.error || response.statusText);
		return payload;
	} catch (error) {
		if (error.name === "AbortError")
			throw new Error(`Request timed out: ${path}`);
		throw error;
	} finally {
		clearTimeout(timeout);
	}
}

function singleFlight(key, operation) {
	if (inFlightRequests.has(key)) return inFlightRequests.get(key);
	const request = Promise.resolve()
		.then(operation)
		.finally(() => inFlightRequests.delete(key));
	inFlightRequests.set(key, request);
	return request;
}

async function loadConfig() {
	try {
		const payload = await api("/api/config");
		if (payload.openai_base_url) {
			$("openaiBaseUrl").value = payload.openai_base_url;
		}
		state.benchmarkProfiles = payload.benchmark_profiles || [];
		renderBenchmarkProfiles();
	} catch (_error) {
		// Keep the static localhost fallback if config cannot be loaded.
	}
}

function profileDisplayLabel(profile) {
	if (!profile) return "Custom";
	if (profile.custom || !profile.version) return profile.label || "Custom";
	return `${profile.label} v${profile.version}`;
}

function benchmarkProfileForRecord(record) {
	if (record?.benchmark_profile) return record.benchmark_profile;
	if (record?.profile_id) {
		return {
			id: record.profile_id,
			label: record.profile_label || record.profile_id,
			version: record.profile_version,
			custom: record.profile_id === "custom",
		};
	}
	return {
		id: "custom",
		label: "Custom (legacy)",
		version: null,
		custom: true,
	};
}

function recordProfileId(record) {
	if (recordSuite(record) === "swe_mini") return "swe_mini";
	return benchmarkProfileForRecord(record).id || "custom";
}

function recordProfileLabel(record) {
	if (recordSuite(record) === "swe_mini") return "SWE Mini";
	return profileDisplayLabel(benchmarkProfileForRecord(record));
}

function normalizeProfileOption(key, value) {
	if (PROFILE_BOOLEAN_OPTIONS.has(key)) return Boolean(value);
	if (PROFILE_NUMERIC_OPTIONS.has(key)) {
		if (value === "" || value === null || value === undefined) return null;
		const parsed = Number(value);
		return Number.isFinite(parsed) ? parsed : String(value);
	}
	if (key === "batch_size") return String(value || "1");
	if (key === "predict_only") return Boolean(value);
	return value;
}

function currentProfileSettings() {
	const settings = { predict_only: false };
	for (const [key, controlId] of Object.entries(PROFILE_OPTION_CONTROLS)) {
		const control = $(controlId);
		const value = PROFILE_BOOLEAN_OPTIONS.has(key)
			? control.checked
			: control.value;
		settings[key] = normalizeProfileOption(key, value);
	}
	return settings;
}

function activeBenchmarkProfile() {
	if (state.activeSuite !== "lm_eval") return null;
	const selectedTasks = [...state.selectedTasks].sort();
	const currentSettings = currentProfileSettings();
	return (
		state.benchmarkProfiles.find((profile) => {
			const expectedTasks = [...(profile.tasks || [])].sort();
			if (JSON.stringify(selectedTasks) !== JSON.stringify(expectedTasks)) {
				return false;
			}
			const expectedSettings = profile.settings || {};
			return Object.keys(expectedSettings).every(
				(key) =>
					normalizeProfileOption(key, currentSettings[key]) ===
					normalizeProfileOption(key, expectedSettings[key]),
			);
		}) || null
	);
}

function updateBenchmarkProfileIndicator() {
	const profile = activeBenchmarkProfile();
	const indicator = $("activeBenchmarkProfile");
	indicator.textContent = `Profile: ${profileDisplayLabel(profile)}`;
	indicator.classList.toggle("custom", !profile);
	for (const profileButton of $("lmEvalProfileButtons").querySelectorAll(
		"button[data-profile-id]",
	)) {
		const isActive = profileButton.dataset.profileId === profile?.id;
		profileButton.classList.toggle("active", isActive);
		profileButton.setAttribute("aria-pressed", String(isActive));
	}
}

function applyBenchmarkProfile(profile) {
	if (profile.warning && !window.confirm(profile.warning)) return;
	state.selectedTasks = new Set(profile.tasks || []);
	state.hasAutoSelectedTask = true;
	$("taskViewMode").value = "leaves";
	for (const [key, expected] of Object.entries(profile.settings || {})) {
		const controlId = PROFILE_OPTION_CONTROLS[key];
		if (!controlId) continue;
		const control = $(controlId);
		if (PROFILE_BOOLEAN_OPTIONS.has(key)) {
			control.checked = Boolean(expected);
		} else {
			control.value =
				expected === null || expected === undefined ? "" : String(expected);
		}
	}
	state.taskPage = 0;
	renderTasks();
	$("setupMessage").textContent =
		`Applied ${profileDisplayLabel(profile)}: ${(profile.tasks || []).length} tasks`;
}

function renderBenchmarkProfiles() {
	const container = $("lmEvalProfileButtons");
	container.replaceChildren();
	state.benchmarkProfiles.forEach((profile) => {
		const limit = profile.settings?.limit;
		const suffix = limit ? `${limit}/task` : "all samples";
		const profileButton = button(`${profile.label} · ${suffix}`);
		profileButton.dataset.profileId = profile.id;
		profileButton.setAttribute("aria-pressed", "false");
		profileButton.title = [profile.description, profile.warning]
			.filter(Boolean)
			.join(" ");
		profileButton.addEventListener("click", () => applyBenchmarkProfile(profile));
		container.append(profileButton);
	});
	updateBenchmarkProfileIndicator();
}

async function loadModels() {
	const base = encodeURIComponent($("openaiBaseUrl").value.trim());
	setText($("modelList"), "Loading models…");
	try {
		const payload = await api(`/api/models?base_url=${base}`);
		state.models = payload.models || [];
		renderModels();
		renderSweJudgeModels();
		renderResults();
	} catch (error) {
		setText($("modelList"), `Could not load models: ${error.message}`);
	}
}
async function loadTasks() {
	const requestedSuite = state.activeSuite;
	const loadToken = ++state.taskLoadToken;
	state.visibleTaskNames = [];
	setTaskLoading(true);
	$("selectVisibleTasks").disabled = true;
	$("unselectVisibleTasks").disabled = true;
	setText(
		$("taskList"),
		`Loading ${suiteLabel(requestedSuite)} ${suiteWorkItems(requestedSuite)}…`,
	);
	try {
		const suite = encodeURIComponent(requestedSuite);
		const payload = await api(`/api/tasks?suite=${suite}`, {
			timeoutMs: TASK_REQUEST_TIMEOUT_MS,
		});
		if (loadToken !== state.taskLoadToken || requestedSuite !== state.activeSuite)
			return;
		state.tasks = payload.tasks || [];
		renderTasks();
	} catch (error) {
		if (loadToken !== state.taskLoadToken || requestedSuite !== state.activeSuite)
			return;
		setText($("taskList"), `Could not load tasks: ${error.message}`);
	} finally {
		if (
			loadToken === state.taskLoadToken &&
			requestedSuite === state.activeSuite
		) {
			setTaskLoading(false);
		}
	}
}
async function loadJobs({ refreshResultsOnTransition = true } = {}) {
	return singleFlight("jobs", async () => {
		const previousStatuses = new Map(
			state.jobs.map((job) => [job.id, job.status]),
		);
		const payload = await api("/api/jobs");
		state.jobs = payload.jobs || [];
		const existing = new Set(state.jobs.map((job) => job.id));
		for (const jobId of state.jobDetails.keys()) {
			if (!existing.has(jobId)) state.jobDetails.delete(jobId);
		}
		for (const jobId of state.jobCommands.keys()) {
			if (!existing.has(jobId)) state.jobCommands.delete(jobId);
		}
		for (const jobId of state.jobLogs.keys()) {
			if (!existing.has(jobId)) state.jobLogs.delete(jobId);
		}
		const reachedTerminalState =
			state.jobsLoaded &&
			state.jobs.some(
				(job) =>
					ACTIVE_JOB_STATUSES.has(previousStatuses.get(job.id)) &&
					TERMINAL_JOB_STATUSES.has(job.status),
			);
		state.jobsLoaded = true;
		renderJobs();
		if (refreshResultsOnTransition && reachedTerminalState) {
			invalidateResultRows();
			void loadResults({ forceRows: true });
		}
		return state.jobs;
	});
}

async function loadLeaderboard() {
	return singleFlight("leaderboard", async () => {
		try {
			const payload = await api("/api/leaderboard", { timeoutMs: 120000 });
			state.leaderboard = payload.leaderboard || [];
			renderLeaderboard();
		} catch (error) {
			setText($("leaderboard"), `Could not load results: ${error.message}`);
		}
	});
}

async function loadResultRows(suite = state.resultSuite, force = false) {
	if (!force && state.loadedResultSuites.has(suite)) {
		renderResults();
		return;
	}
	return singleFlight(`results:${suite}`, async () => {
		setText($("chart"), `Loading ${suiteLabel(suite)} result details…`);
		let offset = 0;
		const rows = [];
		do {
			const query = new URLSearchParams({
				suite,
				offset: String(offset),
				limit: String(RESULT_PAGE_SIZE),
			});
			const payload = await api(`/api/results?${query}`, {
				timeoutMs: 120000,
			});
			rows.push(...(payload.rows || []));
			offset = Number.isFinite(payload.next_offset) ? payload.next_offset : null;
		} while (offset !== null);
		state.rows = [
			...state.rows.filter((row) => recordSuite(row) !== suite),
			...rows,
		];
		state.loadedResultSuites.add(suite);
		renderResults();
	});
}

function invalidateResultRows() {
	state.rows = [];
	state.loadedResultSuites.clear();
}

async function loadResults({ forceRows = false } = {}) {
	await loadLeaderboard();
	if ($("resultDetails").open) {
		try {
			await loadResultRows(state.resultSuite, forceRows);
		} catch (error) {
			setText($("chart"), `Could not load results: ${error.message}`);
		}
	}
}

function renderModels() {
	const list = $("modelList");
	list.replaceChildren();
	if (!state.models.length)
		return setText(list, "No models returned by the OpenAI-compatible endpoint.");
	if (!state.selectedModels.size) state.selectedModels.add(state.models[0].id);
	const filter = $("modelFilter").value.trim().toLowerCase();
	const matchingModels = state.models.filter((model) =>
		`${model.name || model.id} ${(model.labels || []).join(" ")} ${model.recipe || ""}`
			.toLowerCase()
			.includes(filter),
	);
	$("modelCount").textContent =
		`Showing ${matchingModels.length.toLocaleString()} of ${state.models.length.toLocaleString()} models.`;
	matchingModels.forEach((model) => {
		const item = div("item");
		const label = document.createElement("label");
		const checkbox = input("checkbox", "model-choice", model.id);
		checkbox.checked = state.selectedModels.has(model.id);
		checkbox.addEventListener("change", () =>
			checkbox.checked
				? state.selectedModels.add(model.id)
				: state.selectedModels.delete(model.id),
		);
		label.append(
			checkbox,
			summaryBlock(model.name || model.id, modelMeta(model)),
		);
		item.append(label, badgeRow(model.labels || []));
		list.append(item);
	});
}

function renderSweJudgeModels() {
	const select = $("sweJudgeModel");
	const previousValue = select.value || DEFAULT_SWE_JUDGE_MODEL;
	const modelIds = state.models.map((model) => model.id).filter(Boolean);
	select.replaceChildren();
	if (!modelIds.length) {
		select.append(new Option(DEFAULT_SWE_JUDGE_MODEL, DEFAULT_SWE_JUDGE_MODEL));
		select.value = DEFAULT_SWE_JUDGE_MODEL;
		select.disabled = true;
		return;
	}
	select.disabled = false;
	modelIds.forEach((modelId) =>
		select.append(new Option(modelId, modelId, false, false)),
	);
	if (modelIds.includes(previousValue)) {
		select.value = previousValue;
	} else if (modelIds.includes(DEFAULT_SWE_JUDGE_MODEL)) {
		select.value = DEFAULT_SWE_JUDGE_MODEL;
	} else {
		select.value = modelIds[0];
	}
}

function renderTasks() {
	const list = $("taskList");
	list.replaceChildren();
	const filter = $("taskFilter").value.trim().toLowerCase();
	const isLmEval = state.activeSuite === "lm_eval";
	const hideIncompatible = isLmEval && $("hideIncompatibleTasks").checked;
	const hideGated = isLmEval && $("hideGatedTasks").checked;
	const taskViewMode = isLmEval ? $("taskViewMode").value : "leaves";
	if (isLmEval) pruneSelectedTasksForViewMode(taskViewMode);
	const hideNonEnglish = isLmEval && $("hideNonEnglishTasks").checked;
	const selectedCategories = selectedTaskCategories();
	const matchingTasks = state.tasks.filter((task) => {
		if (hideIncompatible && task.compatibility === "incompatible") return false;
		if (hideGated && task.compatibility === "gated") return false;
		if (isLmEval && taskViewMode === "leaves" && (task.kind || "task") !== "task")
			return false;
		if (isLmEval && taskViewMode === "groups" && (task.kind || "task") === "task")
			return false;
		if (hideNonEnglish && task.language_scope === "non_english") return false;
		if (isLmEval && !selectedCategories.has(task.category || "Other"))
			return false;
		return `${task.name} ${task.description || ""} ${task.compatibility || ""} ${task.category || ""} ${task.repo || ""}`
			.toLowerCase()
			.includes(filter);
	});
	if (
		!state.selectedTasks.size &&
		matchingTasks.length &&
		!state.hasAutoSelectedTask
	) {
		const defaults =
			state.activeSuite === "lemonade_bench"
				? matchingTasks.filter((task) => task.default_selected)
				: [matchingTasks[0]];
		(defaults.length ? defaults : [matchingTasks[0]]).forEach((task) =>
			state.selectedTasks.add(task.name),
		);
		state.hasAutoSelectedTask = true;
	}
	renderSelectedTasks();
	const pageCount = Math.max(
		1,
		Math.ceil(matchingTasks.length / TASKS_PER_PAGE),
	);
	state.taskPage = Math.min(state.taskPage, pageCount - 1);
	const renderedTasks = matchingTasks.slice(
		state.taskPage * TASKS_PER_PAGE,
		(state.taskPage + 1) * TASKS_PER_PAGE,
	);
	state.visibleTaskNames = renderedTasks.map((task) => task.name);
	const workItems = suiteWorkItems(state.activeSuite);
	$("taskCount").textContent =
		`Showing ${renderedTasks.length.toLocaleString()} of ${matchingTasks.length.toLocaleString()} matching ${workItems} (${state.tasks.length.toLocaleString()} total).`;
	$("taskPage").textContent = `Page ${state.taskPage + 1} of ${pageCount}`;
	$("taskPrev").disabled = state.taskPage <= 0;
	$("taskNext").disabled = state.taskPage >= pageCount - 1;
	$("selectVisibleTasks").disabled = renderedTasks.length === 0;
	$("unselectVisibleTasks").disabled = renderedTasks.length === 0;
	renderedTasks.forEach((task) => {
		const item = div("item");
		const label = document.createElement("label");
		const checkbox = input("checkbox", "task-choice", task.name);
		checkbox.checked = state.selectedTasks.has(task.name);
		checkbox.addEventListener("change", () => {
			checkbox.checked
				? state.selectedTasks.add(task.name)
				: state.selectedTasks.delete(task.name);
			renderSelectedTasks();
		});
		label.append(checkbox, summaryBlock(task.name, taskMeta(task)));
		item.append(
			label,
			badgeRowNode([
				compatibilityBadge(task.compatibility),
				kindBadge(task.kind),
				categoryBadge(task.category),
			]),
		);
		list.append(item);
	});
}

function setTaskLoading(isLoading) {
	const spinner = $("taskSpinner");
	spinner.hidden = !isLoading;
}
function selectVisibleTasks() {
	state.visibleTaskNames.forEach((taskName) =>
		state.selectedTasks.add(taskName),
	);
	renderTasks();
}
function unselectVisibleTasks() {
	state.visibleTaskNames.forEach((taskName) =>
		state.selectedTasks.delete(taskName),
	);
	renderTasks();
}
function taskMatchesViewMode(task, taskViewMode) {
	const kind = task.kind || "task";
	return taskViewMode === "groups" ? kind !== "task" : kind === "task";
}
function pruneSelectedTasksForViewMode(taskViewMode) {
	if (!state.selectedTasks.size || !state.tasks.length) return;
	const tasksByName = new Map(state.tasks.map((task) => [task.name, task]));
	state.selectedTasks = new Set(
		[...state.selectedTasks].filter((taskName) => {
			const task = tasksByName.get(taskName);
			return task && taskMatchesViewMode(task, taskViewMode);
		}),
	);
}
function renderSelectedTasks() {
	const list = $("selectedTasksList");
	const selected = [...state.selectedTasks].sort((a, b) => a.localeCompare(b));
	$("selectedTaskCount").textContent =
		`${selected.length.toLocaleString()} selected`;
	list.replaceChildren();
	updateBenchmarkProfileIndicator();
	if (!selected.length)
		return setText(list, `No ${suiteWorkItems(state.activeSuite)} selected.`);
	selected.forEach((taskName) => {
		const chip = document.createElement("button");
		chip.className = "selected-chip";
		chip.type = "button";
		chip.textContent = `${taskName} ×`;
		chip.addEventListener("click", () => {
			state.selectedTasks.delete(taskName);
			renderTasks();
		});
		list.append(chip);
	});
}

function visibleJobs() {
	if (state.jobSuiteFilter === "all") return state.jobs;
	return state.jobs.filter((job) => jobSuite(job) === state.jobSuiteFilter);
}

function selectedVisibleJobs() {
	return visibleJobs().filter((job) => state.selectedJobs.has(job.id));
}

function renderJobs() {
	const list = $("jobList");
	const logViews = new Map();
	for (const [jobId, log] of state.jobLogElements) {
		logViews.set(jobId, {
			autoScroll: shouldAutoScrollLog(log),
			scrollTop: log.scrollTop,
		});
	}
	state.jobLogElements.clear();
	list.replaceChildren();
	const existing = new Set(state.jobs.map((job) => job.id));
	state.selectedJobs = new Set(
		[...state.selectedJobs].filter((id) => existing.has(id)),
	);
	state.expandedJobs = new Set(
		[...state.expandedJobs].filter((id) => existing.has(id)),
	);
	const jobs = visibleJobs();
	const visibleLabel = `${jobs.length.toLocaleString()} ${jobs.length === 1 ? "job" : "jobs"}`;
	const totalLabel = `${state.jobs.length.toLocaleString()} ${state.jobs.length === 1 ? "job" : "jobs"}`;
	$("visibleJobCount").textContent =
		state.jobSuiteFilter === "all"
			? visibleLabel
			: `${jobs.length.toLocaleString()} of ${totalLabel}`;
	if (!state.jobs.length) {
		setText(list, "No jobs yet.");
		renderSelectedJobs();
		return;
	}
	if (!jobs.length) {
		setText(list, `No ${suiteLabel(state.jobSuiteFilter)} jobs.`);
		renderSelectedJobs();
		return;
	}
	jobs.forEach((job) => {
		const row = div("job-row");
		const checkbox = input("checkbox", "job-select", job.id);
		checkbox.checked = state.selectedJobs.has(job.id);
		checkbox.addEventListener("click", (event) => event.stopPropagation());
		checkbox.addEventListener("change", () => {
			checkbox.checked
				? state.selectedJobs.add(job.id)
				: state.selectedJobs.delete(job.id);
			renderSelectedJobs();
		});
		const details = document.createElement("details");
		const expanded = div("job-expanded");
		details.className = "job-details";
		details.open = state.expandedJobs.has(job.id);
		details.addEventListener("toggle", () => {
			if (details.open === state.expandedJobs.has(job.id)) return;
			if (details.open) {
				state.expandedJobs.add(job.id);
				renderJobExpanded(expanded, job);
				void loadJobLog(job.id, { forceScroll: true });
			} else {
				state.expandedJobs.delete(job.id);
				state.jobLogElements.delete(job.id);
				expanded.replaceChildren();
			}
		});
		const summary = document.createElement("summary");
		summary.className = "job-summary";
		const summaryActions = div("job-summary-actions");
		const progress = progressBadge(job);
		if (progress) summaryActions.append(progress);
		if (ACTIVE_JOB_STATUSES.has(job.status)) {
			const cancelButton = button(
				job.status === "cancelling" ? "Stopping…" : "Cancel",
				"job-cancel",
			);
			cancelButton.disabled = job.status === "cancelling";
			cancelButton.title = "Stop this job and mark it cancelled";
			cancelButton.addEventListener("click", (event) => {
				event.preventDefault();
				event.stopPropagation();
				void cancelJobs([job.id]);
			});
			summaryActions.append(cancelButton);
		}
		const rerunButton = button("Rerun", "job-rerun");
		rerunButton.disabled = ACTIVE_JOB_STATUSES.has(job.status);
		rerunButton.title = rerunButton.disabled
			? "Cancel this job before rerunning it"
			: "Clear this job and start it again with the same options";
		rerunButton.addEventListener("click", (event) => {
			event.preventDefault();
			event.stopPropagation();
			void rerunJobs([job.id]);
		});
		summaryActions.append(rerunButton);
		const profile = benchmarkProfileBadge(job);
		if (profile) summaryActions.append(profile);
		summaryActions.append(suiteBadge(job), statusBadge(job), checkbox);
		summary.append(
			summaryBlock(
				job.model_id,
				`Job ${job.id} · ${suiteLabel(jobSuite(job))} · ${Number(job.task_count || 0).toLocaleString()} ${suiteWorkItems(jobSuite(job))}`,
			),
			summaryActions,
		);
		if (details.open) renderJobExpanded(expanded, job);
		details.append(summary, expanded);
		row.append(details);
		list.append(row);
	});
	renderSelectedJobs();
	for (const [jobId, view] of logViews) {
		const log = state.jobLogElements.get(jobId);
		if (!log) continue;
		if (view.autoScroll) {
			scrollLogToBottom(log);
		} else {
			log.scrollTop = view.scrollTop;
		}
	}
}

function renderJobExpanded(container, summaryJob) {
	container.replaceChildren();
	const job = state.jobDetails.get(summaryJob.id);
	if (!job) {
		const message = div("job-detail-message");
		setText(message, "Loading job details…");
		container.append(message);
		void loadJobDetails(summaryJob.id);
	} else if (job.detail_error) {
		const message = div("job-detail-message");
		setText(message, `Could not load job details: ${job.detail_error}`);
		container.append(message);
	} else {
		const taskList = document.createElement("ul");
		taskList.className = "job-task-list";
		(job.tasks || []).forEach((taskName) => {
			const taskItem = document.createElement("li");
			taskItem.textContent = taskName;
			taskList.append(taskItem);
		});
		container.append(
			jobDetailMeta({ ...job, ...summaryJob, tasks: job.tasks }),
			taskList,
		);
	}

	const logLabel = div("job-log-label");
	setText(logLabel, "Live output");
	const log = document.createElement("pre");
	log.className = "log job-log";
	log.dataset.jobId = summaryJob.id;
	log.setAttribute("aria-label", `Live output for job ${summaryJob.id}`);
	log.setAttribute("aria-live", "polite");
	log.textContent = state.jobLogs.get(summaryJob.id) || "Loading log output…";
	state.jobLogElements.set(summaryJob.id, log);
	container.append(logLabel, log);
	if (!state.jobLogs.has(summaryJob.id)) {
		void loadJobLog(summaryJob.id, { forceScroll: true });
	}
}

async function loadJobDetails(jobId) {
	if (state.jobDetails.has(jobId)) return state.jobDetails.get(jobId);
	return singleFlight(`job-detail:${jobId}`, async () => {
		try {
			const { job } = await api(`/api/jobs/${jobId}`);
			state.jobDetails.set(jobId, job);
			if (Array.isArray(job.command)) state.jobCommands.set(jobId, job.command);
			if (state.expandedJobs.has(jobId)) renderJobs();
			return job;
		} catch (error) {
			const failedDetail = {
				id: jobId,
				tasks: [],
				detail_error: error.message,
			};
			state.jobDetails.set(jobId, failedDetail);
			if (state.expandedJobs.has(jobId)) renderJobs();
			return failedDetail;
		}
	});
}

function renderSelectedJobs() {
	const selected = selectedVisibleJobs();
	const count = selected.length;
	const activeCount = selected.filter((job) =>
		ACTIVE_JOB_STATUSES.has(job.status),
	).length;
	$("selectedJobCount").textContent = activeCount
		? `${count.toLocaleString()} selected · ${activeCount.toLocaleString()} active`
		: `${count.toLocaleString()} selected`;
	$("cancelSelectedJobs").disabled = activeCount === 0;
	$("clearSelectedJobs").disabled = count === 0 || activeCount > 0;
	$("rerunSelectedJobs").disabled = count === 0 || activeCount > 0;
	syncSelectAllJobs();
}
function syncSelectAllJobs() {
	const checkbox = $("selectAllJobs"),
		jobs = visibleJobs(),
		count = jobs.filter((job) => state.selectedJobs.has(job.id)).length,
		total = jobs.length;
	checkbox.disabled = total === 0;
	checkbox.checked = total > 0 && count === total;
	checkbox.indeterminate = count > 0 && count < total;
}
function toggleAllJobs() {
	const jobs = visibleJobs();
	if ($("selectAllJobs").checked) {
		jobs.forEach((job) => state.selectedJobs.add(job.id));
	} else {
		jobs.forEach((job) => state.selectedJobs.delete(job.id));
	}
	renderJobs();
}
async function loadJobLog(jobId, { forceScroll = false } = {}) {
	return singleFlight(`job-log:${jobId}`, async () => {
		const initialLog = state.jobLogElements.get(jobId);
		const autoScroll =
			forceScroll || !initialLog || shouldAutoScrollLog(initialLog);
		const scrollTop = initialLog?.scrollTop || 0;
		let content;
		try {
			const includeCommand = !state.jobCommands.has(jobId);
			const query = includeCommand ? "?include_command=1" : "";
			const { job } = await api(`/api/jobs/${jobId}/log${query}`);
			if (!state.jobs.some((candidate) => candidate.id === jobId)) return;
			if (Array.isArray(job.command)) state.jobCommands.set(jobId, job.command);
			const command = (state.jobCommands.get(jobId) || []).join(" ");
			content = `$ ${command}\n\n${job.log_tail || "No log output yet."}`;
		} catch (error) {
			if (!state.jobs.some((candidate) => candidate.id === jobId)) return;
			content = error.message;
		}
		state.jobLogs.set(jobId, content);
		const log = state.jobLogElements.get(jobId);
		if (!log) return;
		if (log.textContent !== content) log.textContent = content;
		if (autoScroll) {
			scrollLogToBottom(log);
		} else {
			log.scrollTop = scrollTop;
		}
	});
}
async function loadExpandedJobLogs({
	forceScroll = false,
	includeAll = false,
	includeActive = false,
	includeJobIds = new Set(),
} = {}) {
	const jobs = state.jobs.filter((job) => {
		const isExpanded = state.expandedJobs.has(job.id);
		const needsLiveActivity =
			includeActive && ["running", "cancelling"].includes(job.status);
		return (
			needsLiveActivity ||
			(isExpanded &&
				(includeAll ||
					ACTIVE_JOB_STATUSES.has(job.status) ||
					includeJobIds.has(job.id)))
		);
	});
	await Promise.allSettled(
		jobs.map((job) => loadJobLog(job.id, { forceScroll })),
	);
}
async function refreshJobsAndLogs() {
	await loadJobs();
	await loadExpandedJobLogs({ includeAll: true, includeActive: true });
	renderJobs();
}
function shouldAutoScrollLog(log) {
	return log.scrollHeight - log.scrollTop - log.clientHeight < 24;
}
function scrollLogToBottom(log) {
	log.scrollTop = log.scrollHeight;
}

function renderResultProfileFilter(entries) {
	const select = $("resultProfileFilter");
	const available = new Map(
		state.benchmarkProfiles.map((profile) => [
			profile.id,
			profileDisplayLabel(profile),
		]),
	);
	entries.forEach((entry) =>
		available.set(recordProfileId(entry), recordProfileLabel(entry)),
	);
	select.replaceChildren(new Option("All profiles", "all"));
	for (const [profileId, label] of available) {
		select.append(new Option(label, profileId));
	}
	if (
		[...select.options].some((option) => option.value === state.resultProfile)
	) {
		select.value = state.resultProfile;
	} else {
		state.resultProfile = "all";
		select.value = "all";
	}
}

function renderLeaderboard() {
	const list = $("leaderboard");
	list.replaceChildren();
	const suiteEntries = state.leaderboard.filter(
		(entry) => recordSuite(entry) === state.resultSuite,
	);
	renderResultProfileFilter(suiteEntries);
	const entries =
		state.resultSuite === "lm_eval" && state.resultProfile !== "all"
			? suiteEntries.filter(
					(entry) => recordProfileId(entry) === state.resultProfile,
				)
			: suiteEntries;
	if (!entries.length)
		return setText(
			list,
			`No ${suiteLabel(state.resultSuite)} leaderboard results yet.`,
		);
	if (state.resultSuite === "lemonade_bench") {
		renderLemonadeBenchLeaderboard(list, entries);
		return;
	}
	if (state.resultSuite === "swe_mini") {
		renderSweMiniLeaderboard(list, entries);
		return;
	}
	renderLmEvalLeaderboard(list, entries);
}

function renderLmEvalLeaderboard(list, entries) {
	const profileOrder = new Map(
		state.benchmarkProfiles.map((profile, index) => [profile.id, index]),
	);
	const orderedEntries = [...entries].sort((left, right) => {
		const leftOrder = profileOrder.get(recordProfileId(left)) ?? 999;
		const rightOrder = profileOrder.get(recordProfileId(right)) ?? 999;
		if (leftOrder !== rightOrder) return leftOrder - rightOrder;
		return Number(right.overall_score || 0) - Number(left.overall_score || 0);
	});
	const profileRanks = new Map();
	const rows = orderedEntries.map((entry) => {
		const model = modelForEntry(entry);
		const modelName = entry.model || entry.model_id || "unknown model";
		const profile = benchmarkProfileForRecord(entry);
		let rankNumber = null;
		if (entry.rank_eligible) {
			rankNumber = (profileRanks.get(profile.id) || 0) + 1;
			profileRanks.set(profile.id, rankNumber);
		}
		return { entry, model, modelName, profile, rankNumber };
	});
	const columns = [
		{
			key: "rank",
			label: "#",
			sortLabel: "rank",
			sortValue: (row) => row.rankNumber,
			cell: (row) =>
				leaderboardCell(
					row.rankNumber === null ? "—" : `#${row.rankNumber}`,
					"rank-cell",
				),
		},
		{
			key: "model",
			label: "Model",
			sortValue: (row) => row.modelName,
			cell: (row) => leaderboardCell(row.modelName, "model-cell", row.modelName),
		},
		{
			key: "profile",
			label: "Profile",
			sortValue: (row) => profileDisplayLabel(row.profile),
			cell: (row) =>
				leaderboardCell(
					profileDisplayLabel(row.profile),
					"profile-cell",
					row.profile.id,
				),
		},
		{
			key: "status",
			label: "Status",
			sortValue: (row) => leaderboardStatus(row.entry),
			cell: (row) => leaderboardCell(leaderboardStatus(row.entry)),
		},
		{
			key: "tasks",
			label: "Tasks",
			sortValue: (row) => numberOrNull(row.entry.result_task_count),
			cell: (row) => leaderboardCell(formatTaskCoverage(row.entry)),
		},
		{
			key: "runtime",
			label: "Runtime",
			sortValue: (row) => resultRuntimeSeconds(row.entry),
			cell: (row) =>
				leaderboardCell(
					formatRuntimeSeconds(resultRuntimeSeconds(row.entry)),
					"runtime-cell",
				),
		},
		{
			key: "runtime-backend",
			label: "Backend",
			sortLabel: "Runtime backend",
			sortValue: (row) => modelBackendLabel(row.entry, row.model),
			cell: (row) => leaderboardCell(modelBackendLabel(row.entry, row.model)),
		},
		{
			key: "context",
			label: "Context",
			sortValue: (row) =>
				numberOrNull(row.entry.context_window || row.model?.context_window),
			cell: (row) =>
				leaderboardCell(
					formatContext(row.entry.context_window || row.model?.context_window),
				),
		},
		{
			key: "prompt-rate",
			label: "Prompt tok/s",
			sortValue: (row) => numberOrNull(row.entry.prompt_tok_s),
			cell: (row) => leaderboardCell(formatRate(row.entry.prompt_tok_s)),
		},
		{
			key: "generation-rate",
			label: "Tok/s",
			sortValue: (row) => numberOrNull(row.entry.generation_tok_s),
			cell: (row) => leaderboardCell(formatRate(row.entry.generation_tok_s)),
		},
		{
			key: "ttft",
			label: "TTFT",
			sortValue: (row) => numberOrNull(row.entry.ttft_s),
			cell: (row) => leaderboardCell(formatSeconds(row.entry.ttft_s)),
		},
		{
			key: "overall-score",
			label: "Overall",
			sortLabel: "Balanced Overall",
			sortValue: (row) => numberOrNull(row.entry.overall_score),
			cell: (row) =>
				leaderboardCell(
					formatScore(row.entry.overall_score),
					"score-cell overall-score",
					"Equal-weight mean of reasoning, math, instruction following, and structured output",
				),
		},
	];
	LEADERBOARD_CATEGORIES.forEach((category, index) => {
		columns.push({
			key: `category-${index}`,
			label: category === "Coding / Structured Output" ? "Coding" : category,
			sortLabel: category,
			sortValue: (row) => categoryScoreFor(row.entry, category)?.score,
			cell: (row) => {
				const categoryScore = categoryScoreFor(row.entry, category);
				const cell = leaderboardCell(
					formatScore(categoryScore?.score),
					"score-cell category-score",
				);
				if (categoryScore?.tasks?.length)
					cell.title = categoryScore.tasks.join(", ");
				return cell;
			},
		});
	});
	renderLeaderboardTable(list, rows, columns, "lm_eval");
}

function renderLeaderboardTable(list, rows, columns, suite) {
	const table = document.createElement("table");
	table.className = "leaderboard-table";
	const tbody = document.createElement("tbody");
	for (const row of sortLeaderboardRows(rows, columns, suite)) {
		const tr = document.createElement("tr");
		tr.append(...columns.map((column) => column.cell(row)));
		tbody.append(tr);
	}
	table.append(leaderboardTableHead(columns, suite), tbody);
	list.append(table);
}

function leaderboardTableHead(columns, suite) {
	const thead = document.createElement("thead");
	const header = document.createElement("tr");
	const currentSort = state.leaderboardSort[suite];
	for (const column of columns) {
		const active = currentSort?.key === column.key;
		const th = document.createElement("th");
		th.scope = "col";
		if (active) {
			th.setAttribute(
				"aria-sort",
				currentSort.direction === "asc" ? "ascending" : "descending",
			);
		}
		const button = document.createElement("button");
		button.type = "button";
		button.className = "leaderboard-sort";
		button.dataset.sortKey = column.key;
		const sortLabel = column.sortLabel || column.label;
		const nextDirection =
			active && currentSort.direction === "asc" ? "descending" : "ascending";
		button.setAttribute("aria-label", `Sort by ${sortLabel}, ${nextDirection}`);
		button.title = `Sort by ${sortLabel} (${nextDirection})`;
		const label = document.createElement("span");
		label.textContent = column.label;
		const indicator = document.createElement("span");
		indicator.className = "leaderboard-sort-indicator";
		indicator.setAttribute("aria-hidden", "true");
		let indicatorText = "↕";
		if (active) indicatorText = currentSort.direction === "asc" ? "▲" : "▼";
		indicator.textContent = indicatorText;
		button.append(label, indicator);
		button.addEventListener("click", () => setLeaderboardSort(suite, column));
		th.append(button);
		header.append(th);
	}
	thead.append(header);
	return thead;
}

function setLeaderboardSort(suite, column) {
	const currentSort = state.leaderboardSort[suite];
	const direction =
		currentSort?.key === column.key && currentSort.direction === "asc"
			? "desc"
			: "asc";
	state.leaderboardSort[suite] = { key: column.key, direction };
	renderLeaderboard();
	for (const button of $("leaderboard").querySelectorAll(".leaderboard-sort")) {
		if (button.dataset.sortKey !== column.key) continue;
		button.focus();
		break;
	}
}

function sortLeaderboardRows(rows, columns, suite) {
	const currentSort = state.leaderboardSort[suite];
	const column = columns.find((candidate) => candidate.key === currentSort?.key);
	if (!column) return rows;
	return rows
		.map((row, index) => ({ row, index }))
		.sort((left, right) => {
			const comparison = compareLeaderboardValues(
				column.sortValue(left.row),
				column.sortValue(right.row),
				currentSort.direction,
			);
			return comparison || left.index - right.index;
		})
		.map(({ row }) => row);
}

function compareLeaderboardValues(left, right, direction) {
	const leftMissing = isMissingLeaderboardValue(left);
	const rightMissing = isMissingLeaderboardValue(right);
	if (leftMissing !== rightMissing) return leftMissing ? 1 : -1;
	if (leftMissing) return 0;
	const comparison =
		typeof left === "number" && typeof right === "number"
			? left - right
			: String(left).localeCompare(String(right), undefined, {
					numeric: true,
					sensitivity: "base",
				});
	return direction === "desc" ? -comparison : comparison;
}

function isMissingLeaderboardValue(value) {
	return (
		value === null ||
		value === undefined ||
		value === "" ||
		value === "—" ||
		(typeof value === "number" && !Number.isFinite(value))
	);
}

function leaderboardStatus(entry) {
	if (entry.partial && entry.status === "succeeded") return "incomplete";
	return entry.status || (entry.partial ? "partial" : "—");
}

function formatTaskCoverage(entry) {
	const completed = Number(entry.result_task_count);
	const requested = Number(entry.requested_task_count);
	if (Number.isFinite(completed) && Number.isFinite(requested) && requested > 0)
		return `${completed}/${requested}`;
	if (Number.isFinite(completed)) return String(completed);
	return "—";
}

function renderLemonadeBenchLeaderboard(list, entries) {
	const rows = entries.map((entry, index) => ({
		entry,
		modelName: entry.model || entry.model_id || "unknown model",
		rankNumber: index + 1,
	}));
	const columns = [
		{
			key: "rank",
			label: "#",
			sortLabel: "rank",
			sortValue: (row) => row.rankNumber,
			cell: (row) => leaderboardCell(`#${row.rankNumber}`, "rank-cell"),
		},
		{
			key: "model",
			label: "Model",
			sortValue: (row) => row.modelName,
			cell: (row) => leaderboardCell(row.modelName, "model-cell", row.modelName),
		},
		{
			key: "runtime-backend",
			label: "Backend",
			sortLabel: "Runtime backend",
			sortValue: (row) => row.entry.configuration,
			cell: (row) =>
				leaderboardCell(
					[row.entry.recipe, row.entry.provider_backend].filter(Boolean).join("/") ||
						"unknown",
					"",
					row.entry.backend_args || "",
				),
		},
		{
			key: "context",
			label: "Context",
			sortValue: (row) => numberOrNull(row.entry.context_window),
			cell: (row) => leaderboardCell(formatContext(row.entry.context_window)),
		},
		{
			key: "scenarios",
			label: "Scenarios",
			sortValue: (row) => numberOrNull(row.entry.successful_scenarios),
			cell: (row) =>
				leaderboardCell(
					`${row.entry.successful_scenarios ?? 0}/${row.entry.scenario_count ?? 0}`,
				),
		},
		{
			key: "ttft",
			label: "Avg TTFT",
			sortValue: (row) => numberOrNull(row.entry.average_ttft_ms),
			cell: (row) => leaderboardCell(formatDurationMs(row.entry.average_ttft_ms)),
		},
		{
			key: "generation-rate",
			label: "Avg tok/s",
			sortValue: (row) => numberOrNull(row.entry.average_tps),
			cell: (row) =>
				leaderboardCell(formatRate(row.entry.average_tps), "score-cell"),
		},
		{
			key: "vram",
			label: "Peak VRAM",
			sortValue: (row) => numberOrNull(row.entry.vram_peak_gb),
			cell: (row) => leaderboardCell(formatGigabytes(row.entry.vram_peak_gb)),
		},
		{
			key: "memory",
			label: "Peak RAM",
			sortValue: (row) => numberOrNull(row.entry.memory_peak_gb),
			cell: (row) => leaderboardCell(formatGigabytes(row.entry.memory_peak_gb)),
		},
		{
			key: "failed-runs",
			label: "Failed",
			sortValue: (row) => numberOrNull(row.entry.failed_runs),
			cell: (row) => leaderboardCell(String(row.entry.failed_runs ?? 0)),
		},
		{
			key: "runtime",
			label: "Runtime",
			sortValue: (row) => resultRuntimeSeconds(row.entry),
			cell: (row) =>
				leaderboardCell(
					formatRuntimeSeconds(resultRuntimeSeconds(row.entry)),
					"runtime-cell",
				),
		},
	];
	renderLeaderboardTable(list, rows, columns, "lemonade_bench");
}

function renderSweMiniLeaderboard(list, entries) {
	const rows = entries.map((entry, index) => ({
		entry,
		model: modelForEntry(entry),
		modelName: entry.model || entry.model_id || "unknown model",
		rankNumber: index + 1,
	}));
	const columns = [
		{
			key: "rank",
			label: "#",
			sortLabel: "rank",
			sortValue: (row) => row.rankNumber,
			cell: (row) => leaderboardCell(`#${row.rankNumber}`, "rank-cell"),
		},
		{
			key: "model",
			label: "Model",
			sortValue: (row) => row.modelName,
			cell: (row) => leaderboardCell(row.modelName, "model-cell", row.modelName),
		},
		{
			key: "runtime-backend",
			label: "Backend",
			sortLabel: "Runtime backend",
			sortValue: (row) => modelBackendLabel(row.entry, row.model),
			cell: (row) => leaderboardCell(modelBackendLabel(row.entry, row.model)),
		},
		{
			key: "judge",
			label: "Judge",
			sortValue: (row) => displayJudgeModel(row.entry.judge_model),
			cell: (row) => leaderboardCell(displayJudgeModel(row.entry.judge_model)),
		},
		{
			key: "context",
			label: "Context",
			sortValue: (row) => numberOrNull(row.entry.context_window),
			cell: (row) => leaderboardCell(formatContext(row.entry.context_window)),
		},
		{
			key: "max-output",
			label: "Max output",
			sortValue: (row) => numberOrNull(row.entry.max_output_tokens),
			cell: (row) => leaderboardCell(formatContext(row.entry.max_output_tokens)),
		},
		{
			key: "passed",
			label: "Passed",
			sortValue: (row) => numberOrNull(row.entry.passed_tasks),
			cell: (row) =>
				leaderboardCell(
					`${row.entry.passed_tasks ?? 0}/${row.entry.total_tasks ?? 0}`,
				),
		},
		{
			key: "runtime",
			label: "Runtime",
			sortValue: (row) => resultRuntimeSeconds(row.entry),
			cell: (row) =>
				leaderboardCell(
					formatRuntimeSeconds(resultRuntimeSeconds(row.entry)),
					"runtime-cell",
				),
		},
		{
			key: "success",
			label: "Success",
			sortValue: (row) => numberOrNull(row.entry.overall_score),
			cell: (row) =>
				leaderboardCell(
					formatScore(row.entry.overall_score),
					"score-cell overall-score",
				),
		},
		{
			key: "average-duration",
			label: "Avg duration",
			sortValue: (row) => numberOrNull(row.entry.average_duration_ms),
			cell: (row) =>
				leaderboardCell(formatDurationMs(row.entry.average_duration_ms)),
		},
	];
	renderLeaderboardTable(list, rows, columns, "swe_mini");
}

function detailFilterState() {
	const profile = state.resultSuite === "lm_eval" ? state.resultProfile : "all";
	const contextKey = `${state.resultSuite}:${profile}`;
	if (!state.detailFilters.has(contextKey)) {
		state.detailFilters.set(contextKey, {
			models: new Set(),
			tasks: new Set(),
			metrics: new Set(),
			metricsInitialized: false,
			selectAll: { models: true, tasks: true, metrics: false },
		});
	}
	return state.detailFilters.get(contextKey);
}
function detailFilterValues(rows, key) {
	return [
		...new Set(
			rows.map((row) => String(row[key] ?? "")).filter((value) => value),
		),
	].sort((left, right) =>
		left.localeCompare(right, undefined, { numeric: true, sensitivity: "base" }),
	);
}
function syncDetailFilterSelection(filterState, kind, values) {
	if (!filterState.selectAll[kind]) return;
	filterState[kind].clear();
	values.forEach((value) => filterState[kind].add(value));
}
function createDetailFilterOption(kind, value, labelText, checked, selectAll) {
	const option = document.createElement("label");
	option.className = `detail-filter-option${selectAll ? " all" : ""}`;
	option.title = labelText;
	const checkbox = document.createElement("input");
	checkbox.type = "checkbox";
	checkbox.value = value;
	checkbox.checked = checked;
	checkbox.dataset.detailFilterKind = kind;
	if (selectAll) checkbox.dataset.selectAll = "true";
	const label = document.createElement("span");
	label.textContent = labelText;
	option.append(checkbox, label);
	return { option, checkbox };
}
function renderDetailFilter(kind, values, filterState) {
	const config = DETAIL_FILTER_CONFIG[kind];
	const label =
		kind === "tasks" && state.resultSuite === "lemonade_bench"
			? "Scenarios"
			: config.label;
	const allLabel =
		kind === "tasks" && state.resultSuite === "lemonade_bench"
			? "All scenarios"
			: config.allLabel;
	const container = $(config.optionsId);
	const summary = $(config.summaryId);
	const selectedValues = values.filter((value) => filterState[kind].has(value));
	const allSelected =
		values.length > 0 && selectedValues.length === values.length;
	container.replaceChildren();

	const allOption = createDetailFilterOption(
		kind,
		"",
		allLabel,
		allSelected,
		true,
	);
	allOption.checkbox.indeterminate = selectedValues.length > 0 && !allSelected;
	allOption.checkbox.disabled = values.length === 0;
	container.append(allOption.option);

	values.forEach((value) => {
		container.append(
			createDetailFilterOption(
				kind,
				value,
				value,
				filterState[kind].has(value),
				false,
			).option,
		);
	});

	if (!values.length) {
		summary.textContent = `${label}: None available`;
		summary.title = "";
	} else if (allSelected) {
		summary.textContent = `${label}: All (${values.length})`;
		summary.title = selectedValues.join(", ");
	} else if (selectedValues.length === 1) {
		summary.textContent = `${label}: ${selectedValues[0]}`;
		summary.title = selectedValues[0];
	} else {
		summary.textContent = `${label}: ${selectedValues.length} of ${values.length}`;
		summary.title = selectedValues.join(", ");
	}
}
function handleDetailFilterChange(event) {
	const checkbox = event.target;
	if (!(checkbox instanceof HTMLInputElement)) return;
	const kind = checkbox.dataset.detailFilterKind;
	const config = DETAIL_FILTER_CONFIG[kind];
	if (!config) return;
	const filterState = detailFilterState();
	if (kind === "metrics") filterState.metricsInitialized = true;
	const selection = filterState[kind];
	const optionInputs = [
		...$(config.optionsId).querySelectorAll(
			`input[data-detail-filter-kind="${kind}"]:not([data-select-all])`,
		),
	];
	if (checkbox.dataset.selectAll === "true") {
		checkbox.indeterminate = false;
		filterState.selectAll[kind] = checkbox.checked;
		optionInputs.forEach((option) => {
			option.checked = checkbox.checked;
			if (checkbox.checked) selection.add(option.value);
			else selection.delete(option.value);
		});
	} else {
		if (checkbox.checked) selection.add(checkbox.value);
		else selection.delete(checkbox.value);
		filterState.selectAll[kind] =
			optionInputs.length > 0 && optionInputs.every((option) => option.checked);
	}
	renderResults();
}
function renderResults() {
	renderLeaderboard();
	const suiteRows = state.rows.filter(
		(row) =>
			recordSuite(row) === state.resultSuite &&
			(state.resultSuite !== "lm_eval" ||
				state.resultProfile === "all" ||
				recordProfileId(row) === state.resultProfile),
	);
	const filterState = detailFilterState();
	const models = detailFilterValues(suiteRows, "model");
	const tasks = detailFilterValues(suiteRows, "task");
	syncDetailFilterSelection(filterState, "models", models);
	syncDetailFilterSelection(filterState, "tasks", tasks);
	const metricRows = suiteRows.filter((row) =>
		filterState.tasks.has(String(row.task)),
	);
	const metrics = detailFilterValues(metricRows, "metric");
	if (!filterState.metricsInitialized && metrics.length) {
		const defaultMetric =
			state.resultSuite === "lemonade_bench" && metrics.includes("tps_mean")
				? "tps_mean"
				: metrics[0];
		filterState.metrics.add(defaultMetric);
		filterState.metricsInitialized = true;
	}
	syncDetailFilterSelection(filterState, "metrics", metrics);
	renderDetailFilter("models", models, filterState);
	renderDetailFilter("tasks", tasks, filterState);
	renderDetailFilter("metrics", metrics, filterState);

	const selectedMetrics = metrics.filter((metric) =>
		filterState.metrics.has(metric),
	);
	const rows = suiteRows.filter(
		(row) =>
			filterState.models.has(String(row.model)) &&
			filterState.tasks.has(String(row.task)) &&
			filterState.metrics.has(String(row.metric)),
	);
	renderChart(rows, selectedMetrics);
	renderTable(rows);
}
function renderChart(rows, metrics) {
	const chart = $("chart");
	chart.replaceChildren();
	const metricsWithRows = metrics.filter((metric) =>
		rows.some((row) => row.metric === metric),
	);
	if (!metricsWithRows.length)
		return setText(chart, "No numeric results match the selected filters.");
	metricsWithRows.forEach((metric) => {
		appendMetricChart(
			chart,
			rows.filter((row) => row.metric === metric),
			metric,
		);
	});
}
function appendMetricChart(chart, rows, metric) {
	const group = document.createElement("section");
	group.className = "metric-chart";
	const heading = document.createElement("h3");
	heading.className = "metric-chart-title";
	heading.textContent = metric;
	const canvas = document.createElement("div");
	canvas.className = "metric-chart-canvas";
	const rowHeight = 42;
	const height = Math.max(260, rows.length * rowHeight + 50);
	const visibleWidth = Math.max(
		600,
		Math.floor(canvas.clientWidth || chart.clientWidth || 1000),
	);
	const maxValue = Math.max(...rows.map((row) => Math.abs(row.value)), 1);
	const svg = document.createElementNS(SVG_NS, "svg");
	svg.setAttribute("viewBox", `0 0 ${visibleWidth} ${height}`);
	svg.setAttribute("role", "img");
	svg.setAttribute("aria-label", `${metric} chart`);
	canvas.append(svg);
	group.append(heading, canvas);
	chart.append(group);
	const labels = rows.map((row, index) => {
		const label = svgText(
			10,
			30 + index * rowHeight + 16,
			`${row.model} · ${resultConfigurationLabel(row)} · ${row.task}`,
			"bar-label",
		);
		svg.append(label);
		return label;
	});

	const longestLabelWidth = Math.max(...labels.map(svgTextWidth));
	const barStart = Math.ceil(10 + longestLabelWidth + 24);
	const valueSpace = 90;
	const minimumPlotWidth = 300;
	const width = Math.max(visibleWidth, barStart + minimumPlotWidth + valueSpace);
	const maximumBarWidth = width - barStart - valueSpace;
	svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
	svg.setAttribute("width", String(width));
	svg.setAttribute("height", String(height));
	svg.style.width = `${width}px`;

	rows.forEach((row, index) => {
		const y = 30 + index * rowHeight;
		const barWidth = Math.max(
			2,
			(Math.abs(row.value) / maxValue) * maximumBarWidth,
		);
		svg.append(
			svgRect(barStart, y, barWidth, 24),
			svgText(
				barStart + barWidth + 10,
				y + 16,
				formatValue(row.value),
				"axis-label",
			),
		);
	});
}
function renderTable(rows) {
	const wrap = $("resultTable");
	wrap.replaceChildren();
	if (!rows.length) return;
	const isLemonadeBench = state.resultSuite === "lemonade_bench";
	const headers = isLemonadeBench
		? [
				"Model",
				"Scenario",
				"Category",
				"Backend / context",
				"Metric",
				"Value",
				"Runs",
				"Job",
			]
		: [
				"Model",
				"Profile",
				"Task",
				"Metric",
				"Value",
				"Samples",
				"Runtime",
				"Job",
			];
	const table = document.createElement("table");
	const thead = document.createElement("thead");
	const header = document.createElement("tr");
	headers.forEach((name) => {
		const th = document.createElement("th");
		th.textContent = name;
		header.append(th);
	});
	thead.append(header);
	const tbody = document.createElement("tbody");
	rows.forEach((row) => {
		const tr = document.createElement("tr");
		const values = isLemonadeBench
			? [
					row.model,
					row.task,
					row.scenario_category || "",
					resultConfigurationLabel(row),
					row.metric,
					formatValue(row.value),
					row.samples ?? "",
					row.job_id,
				]
			: [
					row.model,
					recordProfileLabel(row),
					row.task,
					row.metric,
					formatValue(row.value),
					row.samples ?? "",
					formatRuntimeSeconds(resultRuntimeSeconds(row)),
					row.job_id,
				];
		values.forEach((value) => {
			const td = document.createElement("td");
			td.textContent = String(value);
			tr.append(td);
		});
		tbody.append(tr);
	});
	table.append(thead, tbody);
	wrap.append(table);
}
async function cancelSelectedJobs() {
	const jobIds = selectedVisibleJobs()
		.filter((job) => ACTIVE_JOB_STATUSES.has(job.status))
		.map((job) => job.id);
	if (!jobIds.length) return;
	await cancelJobs(jobIds);
}

async function cancelJobs(jobIds) {
	if (!jobIds.length) return;
	$("setupMessage").textContent = "Stopping selected job(s)…";
	try {
		const payload = await api("/api/jobs/cancel", {
			method: "POST",
			body: JSON.stringify({ job_ids: jobIds }),
		});
		state.jobs = payload.jobs || [];
		$("setupMessage").textContent =
			`Cancellation requested for ${payload.cancelled} job(s).`;
		renderJobs();
		await loadExpandedJobLogs({ includeAll: true });
	} catch (error) {
		$("setupMessage").textContent = error.message;
	}
}

async function clearSelectedJobs() {
	const jobIds = selectedVisibleJobs().map((job) => job.id);
	if (!jobIds.length) return;
	try {
		const payload = await api("/api/jobs/clear", {
			method: "POST",
			body: JSON.stringify({ job_ids: jobIds }),
		});
		state.jobs = payload.jobs || [];
		jobIds.forEach((jobId) => state.selectedJobs.delete(jobId));
		state.jobDetails.clear();
		state.jobCommands.clear();
		state.jobLogs.clear();
		state.jobLogElements.clear();
		$("setupMessage").textContent = `Cleared ${payload.cleared} selected job(s).`;
		renderJobs();
		invalidateResultRows();
		await loadResults({ forceRows: true });
	} catch (error) {
		$("setupMessage").textContent = error.message;
	}
}
async function clearFailedJobs() {
	try {
		const payload = await api("/api/jobs/clear-failed", { method: "POST" });
		state.jobs = payload.jobs || [];
		state.selectedJobs.clear();
		state.jobDetails.clear();
		state.jobCommands.clear();
		state.jobLogs.clear();
		state.jobLogElements.clear();
		$("setupMessage").textContent = `Cleared ${payload.cleared} failed job(s).`;
		renderJobs();
		invalidateResultRows();
		await loadResults({ forceRows: true });
	} catch (error) {
		$("setupMessage").textContent = error.message;
	}
}
async function rerunSelectedJobs() {
	const jobIds = selectedVisibleJobs().map((job) => job.id);
	if (!jobIds.length) return;
	await rerunJobs(jobIds);
}

async function rerunJobs(jobIds) {
	if (!jobIds.length) return;
	$("setupMessage").textContent = "Rerunning job…";
	try {
		const payload = await api("/api/jobs/rerun", {
			method: "POST",
			body: JSON.stringify({ job_ids: jobIds }),
		});
		const created = payload.jobs || [];
		const rerunOriginals = created.map((job) => job.rerun_of).filter(Boolean);
		if (rerunOriginals.length) {
			await api("/api/jobs/clear", {
				method: "POST",
				body: JSON.stringify({ job_ids: rerunOriginals }),
			});
		}
		jobIds.forEach((jobId) => state.selectedJobs.delete(jobId));
		if (created.length) state.expandedJobs.add(created.at(-1).id);
		$("setupMessage").textContent = `Started ${created.length} rerun job(s).`;
		await loadJobs({ refreshResultsOnTransition: false });
		invalidateResultRows();
		await loadResults({ forceRows: true });
		await loadExpandedJobLogs({ forceScroll: true, includeAll: true });
	} catch (error) {
		$("setupMessage").textContent = error.message;
	}
}
async function startJobs() {
	const suite = state.activeSuite;
	const modelIds = [...state.selectedModels],
		tasks = [...state.selectedTasks];
	if (!modelIds.length || !tasks.length)
		return ($("setupMessage").textContent =
			`Select at least one model and one ${suiteLabel(suite)} ${suiteWorkItem(suite)}.`);
	const body = {
		suite,
		model_ids: modelIds,
		tasks,
		openai_base_url: $("openaiBaseUrl").value.trim(),
		max_concurrent_jobs:
			suite === "lemonade_bench" ? 1 : Number($("maxConcurrentJobs").value || 1),
	};
	if (suite === "lm_eval") {
		body.llamacpp_backend = $("llamacppBackend").value || null;
	}
	if (suite === "lemonade_bench") {
		const customModels = new Set(
			state.models
				.filter((model) => (model.labels || []).includes("custom"))
				.map((model) => model.id),
		);
		Object.assign(body, {
			lemonade_model_ids: Object.fromEntries(
				modelIds.map((modelId) => [
					modelId,
					customModels.has(modelId) ? `user.${modelId}` : modelId,
				]),
			),
			bench_backends: splitOptionValues($("benchBackends").value),
			bench_context_sizes: splitOptionValues($("benchContextSizes").value)
				.map(Number)
				.filter((value) => Number.isInteger(value) && value > 0),
			bench_runs: Number($("benchRuns").value || DEFAULT_LEMONADE_BENCH_RUNS),
			bench_warmup: Number($("benchWarmup").value || 0),
			bench_timeout: Number(
				$("benchTimeout").value || DEFAULT_LEMONADE_BENCH_TIMEOUT,
			),
			bench_memory_tracking: $("benchMemoryTracking").checked,
			bench_reload_between_runs: $("benchReloadBetweenRuns").checked,
			bench_log_responses: $("benchLogResponses").checked,
		});
	} else if (suite === "swe_mini") {
		const contextWindow =
			numberOrNull($("sweContextWindow").value) || DEFAULT_SWE_CONTEXT_WINDOW;
		const maxOutputTokens =
			numberOrNull($("sweMaxOutputTokens").value) ||
			DEFAULT_SWE_MAX_OUTPUT_TOKENS;
		if (maxOutputTokens > contextWindow) {
			$("setupMessage").textContent =
				"Maximum output tokens cannot exceed the context window.";
			return;
		}
		Object.assign(body, {
			judge_model: $("sweJudgeModel").value.trim() || DEFAULT_SWE_JUDGE_MODEL,
			swe_timeout: Number($("sweTimeout").value || DEFAULT_SWE_TIMEOUT_MINUTES),
			pass_count: Number($("swePassCount").value || 1),
			context_window: contextWindow,
			max_output_tokens: maxOutputTokens,
			swe_provider_timeout: Number(
				$("sweProviderTimeout").value || DEFAULT_SWE_PROVIDER_TIMEOUT_MINUTES,
			),
			recipe_policy: "lemonade_unchanged",
		});
	} else {
		Object.assign(body, {
			limit: $("limit").value.trim() || null,
			num_fewshot:
				$("numFewshot").value === "" ? null : Number($("numFewshot").value),
			max_gen_toks: Number($("maxGenToks").value),
			timeout: Number($("timeout").value),
			num_concurrent: Number($("numConcurrent").value),
			batch_size: $("batchSize").value.trim() || "1",
			task_batch_size: numberOrNull($("taskBatchSize").value),
			apply_chat_template: $("applyChatTemplate").checked,
			fewshot_as_multiturn: $("fewshotAsMultiturn").checked,
			log_samples: $("logSamples").checked,
		});
	}
	$("setupMessage").textContent = "Starting…";
	try {
		const payload = await api("/api/jobs", {
			method: "POST",
			body: JSON.stringify(body),
		});
		$("setupMessage").textContent = `Started ${payload.jobs.length} job(s).`;
		state.resultSuite = suite;
		updateSuiteUi();
		await loadJobs();
		if (suite === "swe_mini" && state.activeSuite === suite) await loadTasks();
	} catch (error) {
		$("setupMessage").textContent = error.message;
	}
}

function statusBadge(job) {
	const status = document.createElement("span");
	status.className = `status ${job.status}`;
	status.textContent = job.status;
	return status;
}
function progressBadge(job) {
	const text = progressText(job);
	if (!text) return null;
	const badge = document.createElement("span");
	badge.className = `badge progress${ACTIVE_JOB_STATUSES.has(job.status) ? " live" : ""}`;
	badge.textContent = text;
	if (ACTIVE_JOB_STATUSES.has(job.status)) {
		badge.title = "Live job activity; refreshed every five seconds";
	}
	return badge;
}
function progressValue(progress) {
	if (!progress || !progress.total) return "";
	const current = Number(progress.current || 0),
		total = Number(progress.total || 0),
		percent = Number(progress.percent || 0);
	if (!Number.isFinite(current) || !Number.isFinite(total) || total <= 0)
		return "";
	const formattedPercent = Number.isFinite(percent)
		? percent.toLocaleString(undefined, { maximumFractionDigits: 1 })
		: "0";
	return `${current}/${total} (${formattedPercent}%)`;
}

function requestProgressFromLog(job) {
	if (!["running", "cancelling"].includes(job.status)) return null;
	const content = state.jobLogs.get(job.id) || "";
	const pattern = /Requesting API:[^\r\n]*?\|\s*(\d+)\/(\d+)\s*\[/g;
	let latest = null;
	for (const match of content.matchAll(pattern)) latest = match;
	if (!latest) return null;
	const current = Number(latest[1]);
	const total = Number(latest[2]);
	if (!Number.isFinite(current) || !Number.isFinite(total) || total <= 0) {
		return null;
	}
	return {
		current,
		total,
		completed: current,
		unit: "requests",
		percent: (current / total) * 100,
	};
}

function durationText(value) {
	if (value === null || value === undefined || value === "") return "";
	const parsed = Number(value);
	if (!Number.isFinite(parsed) || parsed < 0) return "";
	const seconds = Math.floor(parsed);
	if (seconds < 60) return `${seconds}s`;
	const minutes = Math.floor(seconds / 60);
	if (minutes < 60) return `${minutes}m`;
	const hours = Math.floor(minutes / 60);
	const remainingMinutes = minutes % 60;
	if (hours < 24) return `${hours}h ${remainingMinutes}m`;
	const days = Math.floor(hours / 24);
	const remainingHours = hours % 24;
	return `${days}d ${remainingHours}h`;
}

function activeJobElapsed(job) {
	if (!ACTIVE_JOB_STATUSES.has(job.status)) return "";
	const startedAt = Number(job.started_at || job.updated_at || job.created_at);
	if (!Number.isFinite(startedAt) || startedAt <= 0) return "";
	return durationText(Date.now() / 1000 - startedAt);
}

function completedJobRuntime(job) {
	if (!TERMINAL_JOB_STATUSES.has(job.status)) return "";
	const persisted = durationText(job.runtime_seconds);
	if (persisted) return persisted;
	const startedAt = Number(job.started_at);
	const finishedAt = Number(job.finished_at || job.updated_at);
	if (
		!Number.isFinite(startedAt) ||
		startedAt <= 0 ||
		!Number.isFinite(finishedAt) ||
		finishedAt < startedAt
	)
		return "";
	return durationText(finishedAt - startedAt);
}

function withElapsed(text, elapsed) {
	return elapsed ? `${text} · ${elapsed}` : text;
}

function progressText(job) {
	const runtime = completedJobRuntime(job);
	if (runtime) return `Runtime ${runtime}`;
	const elapsed = activeJobElapsed(job);
	const requestProgress = job.request_progress || requestProgressFromLog(job);
	const requestValue = progressValue(requestProgress);
	if (requestValue) return withElapsed(`${requestValue} requests`, elapsed);

	const progress = job.progress;
	const value = progressValue(progress);
	if (value) {
		const current = Number(progress.current || 0);
		const completed = Number(progress.completed || 0);
		if (progress.unit === "batches" && current > completed) {
			return withElapsed(`Batch ${current}/${Number(progress.total)}`, elapsed);
		}
		return withElapsed(`${value} ${progress.unit || "items"}`, elapsed);
	}
	if (job.status === "running") return withElapsed("Running", elapsed);
	if (job.status === "queued") return withElapsed("Waiting", elapsed);
	if (job.status === "cancelling") return withElapsed("Stopping", elapsed);
	return "";
}
function benchmarkProfileBadge(job) {
	if (jobSuite(job) !== "lm_eval") return null;
	const profile = benchmarkProfileForRecord(job);
	const badge = document.createElement("span");
	badge.className = `badge profile${profile.custom ? " custom" : ""}`;
	badge.textContent = profileDisplayLabel(profile);
	badge.title = profile.id || "custom";
	return badge;
}
function suiteBadge(job) {
	const badge = document.createElement("span");
	badge.className = "badge suite";
	badge.textContent = suiteLabel(jobSuite(job));
	return badge;
}
function jobDetailMeta(job) {
	const details = div("job-meta");
	const options = job.swe_options || {};
	const benchOptions = job.lemonade_bench_options || {};
	const evalOptions = job.eval_options || {};
	const progress = job.progress || {};
	const requestProgress =
		job.request_progress || requestProgressFromLog(job) || {};
	const batchProgress = job.batch_progress || {};
	const currentTasks = Array.isArray(batchProgress.current_tasks)
		? batchProgress.current_tasks
		: [];
	const currentTaskLabel =
		currentTasks.length === 1 ? "Current task" : "Current tasks";
	const protection = job.model_protection || {};
	const values = [
		`Suite: ${suiteLabel(jobSuite(job))}`,
		jobSuite(job) === "lm_eval" ? `Profile: ${recordProfileLabel(job)}` : null,
		ACTIVE_JOB_STATUSES.has(job.status) &&
		progress.unit !== "batches" &&
		progressText(job)
			? `Progress: ${progressText(job)}`
			: null,
		batchProgress.current
			? `Current task batch: ${batchProgress.current}/${batchProgress.total}`
			: null,
		currentTasks.length
			? `${currentTaskLabel}: ${currentTasks.join(", ")}`
			: null,
		progressValue(requestProgress)
			? `Current batch requests: ${progressValue(requestProgress)}`
			: null,
		activeJobElapsed(job) ? `Elapsed: ${activeJobElapsed(job)}` : null,
		completedJobRuntime(job) ? `Runtime: ${completedJobRuntime(job)}` : null,
		job.rerun_of ? `Rerun of: ${job.rerun_of}` : null,
		evalOptions.task_batch_size
			? `Task batch size: ${evalOptions.task_batch_size}`
			: null,
		batchProgress.total
			? `Completed task batches: ${batchProgress.completed || 0}/${batchProgress.total}`
			: null,
		protection.state ? `Model protection: ${protection.state}` : null,
		options.judge_model
			? `Judge: ${displayJudgeModel(options.judge_model)}`
			: null,
		options.pass_count ? `Pass attempts: ${options.pass_count}` : null,
		options.context_window
			? `Agent context: ${formatContext(options.context_window)}`
			: null,
		options.max_output_tokens
			? `Max output: ${formatContext(options.max_output_tokens)}`
			: null,
		options.provider_timeout_minutes
			? `Provider timeout: ${options.provider_timeout_minutes}m`
			: null,
		options.provider_max_retries !== undefined
			? `Provider retries: ${options.provider_max_retries}`
			: null,
		options.recipe_policy === "lemonade_unchanged"
			? "Recipe policy: Lemonade unchanged"
			: null,
		jobSuite(job) === "lemonade_bench"
			? `Runs: ${benchOptions.measurement_runs || DEFAULT_LEMONADE_BENCH_RUNS}`
			: null,
		jobSuite(job) === "lemonade_bench"
			? `Warmups: ${benchOptions.warmup_runs || 0}`
			: null,
		jobSuite(job) === "lemonade_bench" && benchOptions.backends?.length
			? `Backends: ${benchOptions.backends.join(", ")}`
			: null,
		jobSuite(job) === "lemonade_bench" && benchOptions.context_sizes?.length
			? `Contexts: ${benchOptions.context_sizes.map(formatContext).join(", ")}`
			: null,
		job.provider_backend ? `Runtime backend: ${job.provider_backend}` : null,
	].filter(Boolean);
	details.textContent = values.join(" · ");
	return details;
}
function jobSuite(job) {
	return recordSuite(job);
}
function recordSuite(record) {
	return record?.suite || "lm_eval";
}
function suiteLabel(suite) {
	return SUITES[suite] || suite || "lm-eval";
}
function suiteWorkItem(suite) {
	return suite === "lemonade_bench" ? "scenario" : "task";
}
function suiteWorkItems(suite) {
	return `${suiteWorkItem(suite)}s`;
}
function splitOptionValues(value) {
	return String(value || "")
		.replaceAll(",", " ")
		.split(/\s+/)
		.map((item) => item.trim())
		.filter(Boolean);
}
function resultConfigurationLabel(record) {
	if (recordSuite(record) === "lemonade_bench") {
		return (
			record.configuration ||
			`${[record.recipe, record.backend].filter(Boolean).join("/") || "default"} · ${formatContext(record.context_window)}`
		);
	}
	return recordProfileLabel(record);
}
function numberOrNull(value) {
	return value === "" || value === null || value === undefined
		? null
		: Number(value);
}
function summaryBlock(title, meta) {
	const span = document.createElement("span"),
		strong = document.createElement("strong"),
		br = document.createElement("br"),
		small = document.createElement("span");
	strong.textContent = title;
	small.className = "meta";
	small.textContent = meta;
	span.append(strong, br, small);
	return span;
}
function badgeRow(labels) {
	const row = document.createElement("div");
	labels.forEach((label) => {
		const badge = document.createElement("span");
		badge.className = "badge";
		badge.textContent = label;
		row.append(badge);
	});
	return row;
}
function modelMeta(model) {
	return [
		model.recipe,
		model.size_gb ? `${model.size_gb} GB` : null,
		model.context_window ? `${model.context_window.toLocaleString()} ctx` : null,
	]
		.filter(Boolean)
		.join(" · ");
}
function taskMeta(task) {
	return task.description || "";
}
function badgeRowNode(nodes) {
	const row = document.createElement("div");
	nodes.filter(Boolean).forEach((node) => row.append(node));
	return row;
}
function compatibilityBadge(compatibility = "unknown") {
	const badge = document.createElement("span");
	badge.className = `badge compatibility ${compatibility}`;
	badge.textContent = compatibility;
	return badge;
}
function categoryBadge(category = "Other") {
	const badge = document.createElement("span");
	badge.className = "badge category";
	badge.textContent = category || "Other";
	return badge;
}
function kindBadge(kind = "task") {
	const badge = document.createElement("span");
	badge.className = "badge kind";
	badge.textContent = kind || "task";
	return badge;
}
function isClientBackend(backend) {
	return CLIENT_BACKENDS.has(String(backend));
}
function specificRuntimeBackend(value) {
	if (!value) return null;
	const backend = String(value);
	return backend === "llamacpp" || isClientBackend(backend) ? null : backend;
}
function recipeBackend(recipe) {
	if (!recipe) return null;
	const backend = String(recipe);
	return backend === "llamacpp" ? "system" : backend;
}
function modelBackendLabel(entry, model) {
	return (
		specificRuntimeBackend(entry.provider_backend) ||
		specificRuntimeBackend(entry.lemonade_backend) ||
		specificRuntimeBackend(entry.llamacpp_backend) ||
		specificRuntimeBackend(entry.requested_llamacpp_backend) ||
		specificRuntimeBackend(entry.runtime_backend) ||
		specificRuntimeBackend(model?.llamacpp_backend) ||
		specificRuntimeBackend(model?.runtime_backend) ||
		recipeBackend(entry.recipe) ||
		recipeBackend(model?.recipe) ||
		specificRuntimeBackend(entry.backend) ||
		"unknown"
	);
}
function modelForEntry(entry) {
	return state.models.find(
		(model) =>
			model.id === entry.model_id ||
			model.id === entry.model ||
			model.name === entry.model,
	);
}
function categoryScoreFor(entry, category) {
	return (entry.category_scores || []).find(
		(score) => score.category === category,
	);
}
function displayJudgeModel(judgeModel) {
	return String(judgeModel || "—").replace(/^lemonade\//, "");
}
function leaderboardCell(value, className = "", title = "") {
	const cell = document.createElement("td");
	if (className) cell.className = className;
	cell.textContent = value ?? "—";
	if (title) cell.title = title;
	return cell;
}
function div(className) {
	const node = document.createElement("div");
	node.className = className;
	return node;
}
function input(type, className, value) {
	const node = document.createElement("input");
	node.type = type;
	node.className = className;
	node.value = value;
	return node;
}
function button(text, className = "") {
	const node = document.createElement("button");
	node.type = "button";
	if (className) node.className = className;
	node.textContent = text;
	return node;
}
function svgRect(x, y, width, height) {
	const rect = document.createElementNS(SVG_NS, "rect");
	rect.setAttribute("x", x);
	rect.setAttribute("y", y);
	rect.setAttribute("width", width);
	rect.setAttribute("height", height);
	rect.setAttribute("rx", "6");
	rect.setAttribute("fill", "#58a6ff");
	return rect;
}
function svgText(x, y, value, className) {
	const text = document.createElementNS(SVG_NS, "text");
	text.setAttribute("x", x);
	text.setAttribute("y", y);
	text.setAttribute("class", className);
	text.textContent = value;
	return text;
}
function svgTextWidth(text) {
	if (typeof text.getComputedTextLength === "function") {
		try {
			const measured = text.getComputedTextLength();
			if (Number.isFinite(measured) && measured > 0) return measured;
		} catch (_error) {
			// Detached or synthetic SVG implementations may not support measurement.
		}
	}
	return String(text.textContent || "").length * 7.5;
}
function setText(node, value) {
	node.replaceChildren();
	node.textContent = value;
}
function formatValue(value) {
	return Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 });
}
function formatScore(value) {
	return value === null || value === undefined || Number.isNaN(Number(value))
		? "—"
		: `${Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 })}%`;
}
function resultRuntimeSeconds(record) {
	const rawRuntime = record?.runtime_seconds;
	const runtime = Number(rawRuntime);
	if (
		rawRuntime !== null &&
		rawRuntime !== undefined &&
		Number.isFinite(runtime) &&
		runtime >= 0
	)
		return runtime;
	const rawEvaluationRuntime = record?.total_evaluation_time_seconds;
	const evaluationRuntime = Number(rawEvaluationRuntime);
	if (
		rawEvaluationRuntime !== null &&
		rawEvaluationRuntime !== undefined &&
		Number.isFinite(evaluationRuntime) &&
		evaluationRuntime >= 0
	) {
		return evaluationRuntime;
	}
	const rawSweRuntimeMs = record?.total_duration_ms;
	const sweRuntimeMs = Number(rawSweRuntimeMs);
	return rawSweRuntimeMs !== null &&
		rawSweRuntimeMs !== undefined &&
		Number.isFinite(sweRuntimeMs) &&
		sweRuntimeMs >= 0
		? sweRuntimeMs / 1000
		: null;
}
function formatRuntimeSeconds(value) {
	return durationText(value) || "—";
}
function formatRate(value) {
	return value === null || value === undefined || Number.isNaN(Number(value))
		? "—"
		: `${Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 })}`;
}
function formatGigabytes(value) {
	return value === null || value === undefined || Number.isNaN(Number(value))
		? "—"
		: `${Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 })} GB`;
}
function formatSeconds(value) {
	return value === null || value === undefined || Number.isNaN(Number(value))
		? "—"
		: `${Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 })}s`;
}
function formatDurationMs(value) {
	return value === null || value === undefined || Number.isNaN(Number(value))
		? "—"
		: formatSeconds(Number(value) / 1000);
}
function formatContext(value) {
	return value === null || value === undefined || Number.isNaN(Number(value))
		? "—"
		: `${Number(value).toLocaleString()} ctx`;
}
function resetTaskPage() {
	state.taskPage = 0;
	renderTasks();
}
function changeTaskPage(delta) {
	state.taskPage = Math.max(0, state.taskPage + delta);
	renderTasks();
}
function selectedTaskCategories() {
	return new Set(
		TASK_CATEGORY_FILTERS.filter(({ id }) => $(id).checked).map(
			({ category }) => category,
		),
	);
}
function updateSuiteUi() {
	const isLemonadeBench = state.activeSuite === "lemonade_bench";
	const isLmEval = state.activeSuite === "lm_eval";
	const isSweMini = state.activeSuite === "swe_mini";
	let taskPlaceholder = "Type to search 14k+ tasks";
	let taskHint =
		"OpenAI-compatible chat backends are generation oriented. Use generate_until tasks first.";
	if (isLemonadeBench) {
		taskPlaceholder = "Type to search benchmark scenarios";
		taskHint =
			"Lemonade Bench measures TTFT, token throughput, request duration, and memory use. Long-context scenarios are opt-in and can run for a long time.";
	} else if (isSweMini) {
		taskPlaceholder = "Type to search SWE Mini tasks or repos";
		taskHint =
			"SWE Mini tasks run in Docker SWE-bench containers and are judged by the selected judge model.";
	}
	let leaderboardDescription =
		"Balanced Overall gives equal weight to reasoning, math, instruction following, and structured output. Rankings are kept separate by profile. Select any column heading to sort; select it again to reverse the order.";
	if (state.resultSuite === "lemonade_bench") {
		leaderboardDescription =
			"Lemonade Bench compares average TTFT, token throughput, duration, and peak memory for each backend/context combination. Select any column heading to sort; select it again to reverse the order.";
	} else if (state.resultSuite === "swe_mini") {
		leaderboardDescription =
			"SWE Mini ranks models by judged task success and shows runtime and average task duration. Select any column heading to sort; select it again to reverse the order.";
	}
	$("taskPanelTitle").textContent =
		`${suiteLabel(state.activeSuite)} ${suiteWorkItems(state.activeSuite)}`;
	$("selectedWorkItemsTitle").textContent =
		`Selected ${suiteWorkItems(state.activeSuite)}`;
	$("selectVisibleTasks").textContent =
		`Select visible ${suiteWorkItems(state.activeSuite)}`;
	$("unselectVisibleTasks").textContent =
		`Unselect visible ${suiteWorkItems(state.activeSuite)}`;
	$("taskFilter").placeholder = taskPlaceholder;
	$("taskViewModeControl").hidden = !isLmEval;
	$("lmEvalProfilePicker").hidden = !isLmEval;
	$("lmEvalCategoryFilters").hidden = !isLmEval;
	$("lmEvalCompatibilityFilters").hidden = !isLmEval;
	$("modelRuntimeOptions").hidden = !isLmEval;
	$("lemonadeBenchOptions").hidden = !isLemonadeBench;
	$("lmEvalBenchmarkOptions").hidden = !isLmEval;
	$("sweMiniBenchmarkOptions").hidden = !isSweMini;
	$("sweMiniJudgeHint").hidden = !isSweMini;
	$("resultProfileControl").hidden = state.resultSuite !== "lm_eval";
	$("taskHint").textContent = taskHint;
	$("leaderboardDescription").textContent = leaderboardDescription;
	$("resultDetailsSummary").textContent =
		`${suiteLabel(state.resultSuite)} detailed metric comparison`;
	$("detailResultsTitle").textContent =
		`${suiteLabel(state.resultSuite)} detailed results`;
	for (const button of [
		$("suiteLemonadeBench"),
		$("suiteLmEval"),
		$("suiteSweMini"),
	]) {
		button.classList.toggle("active", button.dataset.suite === state.activeSuite);
	}
	updateBenchmarkProfileIndicator();
	for (const button of [
		$("leaderboardLemonadeBench"),
		$("leaderboardLmEval"),
		$("leaderboardSweMini"),
	]) {
		button.classList.toggle("active", button.dataset.suite === state.resultSuite);
	}
}
async function selectBenchmarkSuite(suite) {
	if (state.activeSuite === suite) return;
	state.activeSuite = suite;
	state.selectedTasks.clear();
	state.visibleTaskNames = [];
	state.taskPage = 0;
	state.hasAutoSelectedTask = false;
	updateSuiteUi();
	renderSelectedTasks();
	await loadTasks();
}
async function selectResultSuite(suite) {
	state.resultSuite = suite;
	updateSuiteUi();
	renderResults();
	if ($("resultDetails").open) {
		try {
			await loadResultRows(suite);
		} catch (error) {
			setText($("chart"), `Could not load results: ${error.message}`);
		}
	}
}

$("refreshModels").addEventListener("click", loadModels);
$("modelFilter").addEventListener("input", renderModels);
$("jobSuiteFilter").addEventListener("change", () => {
	state.jobSuiteFilter = $("jobSuiteFilter").value;
	renderJobs();
});
$("selectAllJobs").addEventListener("change", toggleAllJobs);
$("cancelSelectedJobs").addEventListener("click", cancelSelectedJobs);
$("clearSelectedJobs").addEventListener("click", clearSelectedJobs);
$("rerunSelectedJobs").addEventListener("click", rerunSelectedJobs);
$("clearFailedJobs").addEventListener("click", clearFailedJobs);
$("refreshJobs").addEventListener("click", () =>
	Promise.all([refreshJobsAndLogs(), loadResults({ forceRows: true })]),
);
$("refreshAll").addEventListener("click", () =>
	Promise.all([
		loadModels(),
		loadTasks(),
		refreshJobsAndLogs(),
		loadResults({ forceRows: true }),
	]),
);
$("resultDetails").addEventListener("toggle", () => {
	if (!$("resultDetails").open) return;
	void loadResultRows(state.resultSuite).catch((error) =>
		setText($("chart"), `Could not load results: ${error.message}`),
	);
});
$("startJobs").addEventListener("click", startJobs);
$("selectVisibleTasks").addEventListener("click", selectVisibleTasks);
$("unselectVisibleTasks").addEventListener("click", unselectVisibleTasks);
$("taskFilter").addEventListener("input", resetTaskPage);
$("hideIncompatibleTasks").addEventListener("change", resetTaskPage);
$("hideGatedTasks").addEventListener("change", resetTaskPage);
$("taskViewMode").addEventListener("change", resetTaskPage);
$("hideNonEnglishTasks").addEventListener("change", resetTaskPage);
TASK_CATEGORY_FILTERS.forEach(({ id }) =>
	$(id).addEventListener("change", resetTaskPage),
);
$("taskPrev").addEventListener("click", () => changeTaskPage(-1));
$("taskNext").addEventListener("click", () => changeTaskPage(1));
Object.values(DETAIL_FILTER_CONFIG).forEach(({ optionsId }) =>
	$(optionsId).addEventListener("change", handleDetailFilterChange),
);
$("resultProfileFilter").addEventListener("change", () => {
	state.resultProfile = $("resultProfileFilter").value;
	renderResults();
});
Object.values(PROFILE_OPTION_CONTROLS).forEach((controlId) => {
	const control = $(controlId);
	control.addEventListener("input", updateBenchmarkProfileIndicator);
	control.addEventListener("change", updateBenchmarkProfileIndicator);
});
$("suiteLemonadeBench").addEventListener("click", () =>
	selectBenchmarkSuite("lemonade_bench"),
);
$("suiteLmEval").addEventListener("click", () =>
	selectBenchmarkSuite("lm_eval"),
);
$("suiteSweMini").addEventListener("click", () =>
	selectBenchmarkSuite("swe_mini"),
);
$("leaderboardLemonadeBench").addEventListener(
	"click",
	() => void selectResultSuite("lemonade_bench"),
);
$("leaderboardLmEval").addEventListener(
	"click",
	() => void selectResultSuite("lm_eval"),
);
$("leaderboardSweMini").addEventListener(
	"click",
	() => void selectResultSuite("swe_mini"),
);

document.addEventListener("visibilitychange", () => {
	if (!document.hidden) scheduleJobPoll(0);
});

function scheduleJobPoll(delay = JOB_POLL_INTERVAL_MS) {
	if (jobPollTimer !== null) clearTimeout(jobPollTimer);
	jobPollTimer = setTimeout(pollJobs, delay);
}

async function pollJobs() {
	jobPollTimer = null;
	if (document.hidden) {
		scheduleJobPoll();
		return;
	}
	try {
		const previouslyActive = new Set(
			state.jobs
				.filter((job) => ACTIVE_JOB_STATUSES.has(job.status))
				.map((job) => job.id),
		);
		await loadJobs();
		jobPollFailures = 0;
		await loadExpandedJobLogs({
			includeActive: true,
			includeJobIds: previouslyActive,
		});
		renderJobs();
	} catch (_error) {
		jobPollFailures += 1;
	}
	const backoff = Math.min(
		60000,
		JOB_POLL_INTERVAL_MS * 2 ** Math.min(jobPollFailures, 4),
	);
	scheduleJobPoll(backoff);
}

async function bootstrap() {
	updateSuiteUi();
	await loadConfig();
	await Promise.allSettled([
		loadModels(),
		loadTasks(),
		loadJobs({ refreshResultsOnTransition: false }),
	]);
	await loadExpandedJobLogs({ includeActive: true });
	renderJobs();
	scheduleJobPoll();
	void loadResults();
}

void bootstrap();

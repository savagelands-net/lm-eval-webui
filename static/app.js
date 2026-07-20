const state = {
	models: [],
	tasks: [],
	jobs: [],
	rows: [],
	leaderboard: [],
	selectedJobId: null,
	selectedJobs: new Set(),
	expandedJobs: new Set(),
	jobDetails: new Map(),
	jobCommands: new Map(),
	loadedResultSuites: new Set(),
	jobsLoaded: false,
	selectedModels: new Set(),
	selectedTasks: new Set(),
	visibleTaskNames: [],
	hasAutoSelectedTask: false,
	taskPage: 0,
	taskLoadToken: 0,
	activeSuite: "lm_eval",
	resultSuite: "lm_eval",
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
	lm_eval: "lm-eval",
	swe_mini: "SWE Mini",
};
const DEFAULT_SWE_JUDGE_MODEL = "gpt-oss-120b-mxfp-GGUF";
const ACTIVE_JOB_STATUSES = new Set(["queued", "running", "cancelling"]);
const TERMINAL_JOB_STATUSES = new Set(["cancelled", "failed", "succeeded"]);
const RESULT_PAGE_SIZE = 2000;
const JOB_POLL_INTERVAL_MS = 5000;
const REQUEST_TIMEOUT_MS = 30000;
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
	} catch (_error) {
		// Keep the static localhost fallback if config cannot be loaded.
	}
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
	setText($("taskList"), `Loading ${suiteLabel(requestedSuite)} tasks…`);
	try {
		const suite = encodeURIComponent(requestedSuite);
		const payload = await api(`/api/tasks?suite=${suite}`);
		if (
			loadToken !== state.taskLoadToken ||
			requestedSuite !== state.activeSuite
		)
			return;
		state.tasks = payload.tasks || [];
		renderTasks();
	} catch (error) {
		if (
			loadToken !== state.taskLoadToken ||
			requestedSuite !== state.activeSuite
		)
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
			offset = Number.isFinite(payload.next_offset)
				? payload.next_offset
				: null;
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
		return setText(
			list,
			"No models returned by the OpenAI-compatible endpoint.",
		);
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
	pruneSelectedTasksForViewMode(taskViewMode);
	const hideNonEnglish = isLmEval && $("hideNonEnglishTasks").checked;
	const selectedCategories = selectedTaskCategories();
	const matchingTasks = state.tasks.filter((task) => {
		if (hideIncompatible && task.compatibility === "incompatible") return false;
		if (hideGated && task.compatibility === "gated") return false;
		if (taskViewMode === "leaves" && (task.kind || "task") !== "task")
			return false;
		if (taskViewMode === "groups" && (task.kind || "task") === "task")
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
		state.selectedTasks.add(matchingTasks[0].name);
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
	$("taskCount").textContent =
		`Showing ${renderedTasks.length.toLocaleString()} of ${matchingTasks.length.toLocaleString()} matching tasks (${state.tasks.length.toLocaleString()} total).`;
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
	if (!selected.length) return setText(list, "No tasks selected.");
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

function renderJobs() {
	const list = $("jobList");
	list.replaceChildren();
	const existing = new Set(state.jobs.map((job) => job.id));
	state.selectedJobs = new Set(
		[...state.selectedJobs].filter((id) => existing.has(id)),
	);
	state.expandedJobs = new Set(
		[...state.expandedJobs].filter((id) => existing.has(id)),
	);
	if (!state.jobs.length) {
		setText(list, "No jobs yet.");
		$("jobLog").textContent = "";
		renderSelectedJobs();
		return;
	}
	state.jobs.forEach((job) => {
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
			if (details.open) {
				state.expandedJobs.add(job.id);
				if (state.jobDetails.has(job.id)) {
					renderJobExpanded(expanded, job);
				} else {
					void loadJobDetails(job.id);
				}
			} else {
				state.expandedJobs.delete(job.id);
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
		summaryActions.append(
			rerunButton,
			suiteBadge(job),
			statusBadge(job),
			checkbox,
		);
		summary.append(
			summaryBlock(
				job.model_id,
				`Job ${job.id} · ${suiteLabel(jobSuite(job))} · ${Number(job.task_count || 0).toLocaleString()} tasks`,
			),
			summaryActions,
		);
		summary.addEventListener("click", () => void selectJob(job.id));
		if (details.open) renderJobExpanded(expanded, job);
		details.append(summary, expanded);
		row.append(details);
		list.append(row);
	});
	renderSelectedJobs();
	if (!state.selectedJobId && state.jobs.length)
		void selectJob(state.jobs.at(-1).id);
}

function renderJobExpanded(container, summaryJob) {
	container.replaceChildren();
	const job = state.jobDetails.get(summaryJob.id);
	if (!job) {
		setText(container, "Expand to load job details…");
		return;
	}
	if (job.detail_error) {
		setText(container, `Could not load job details: ${job.detail_error}`);
		return;
	}
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
	const selected = state.jobs.filter((job) => state.selectedJobs.has(job.id));
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
		count = state.selectedJobs.size,
		total = state.jobs.length;
	checkbox.disabled = total === 0;
	checkbox.checked = total > 0 && count === total;
	checkbox.indeterminate = count > 0 && count < total;
}
function toggleAllJobs() {
	if ($("selectAllJobs").checked) {
		state.selectedJobs = new Set(state.jobs.map((job) => job.id));
	} else {
		state.selectedJobs.clear();
	}
	renderJobs();
}
async function selectJob(jobId) {
	state.selectedJobId = jobId;
	await loadSelectedJobLog({ forceScroll: true });
}
async function loadSelectedJobLog({ forceScroll = false } = {}) {
	const jobId = state.selectedJobId;
	if (!jobId) return;
	return singleFlight(`job-log:${jobId}`, async () => {
		const log = $("jobLog");
		const autoScroll = forceScroll || shouldAutoScrollLog(log);
		try {
			const includeCommand = !state.jobCommands.has(jobId);
			const query = includeCommand ? "?include_command=1" : "";
			const { job } = await api(`/api/jobs/${jobId}/log${query}`);
			if (state.selectedJobId !== jobId) return;
			if (Array.isArray(job.command)) state.jobCommands.set(jobId, job.command);
			const command = (state.jobCommands.get(jobId) || []).join(" ");
			const content = `$ ${command}\n\n${job.log_tail || "No log output yet."}`;
			if (log.textContent !== content) log.textContent = content;
		} catch (error) {
			if (state.selectedJobId === jobId) log.textContent = error.message;
		}
		if (autoScroll && state.selectedJobId === jobId) scrollLogToBottom(log);
	});
}
function shouldAutoScrollLog(log) {
	return log.scrollHeight - log.scrollTop - log.clientHeight < 24;
}
function scrollLogToBottom(log) {
	log.scrollTop = log.scrollHeight;
}

function renderLeaderboard() {
	const list = $("leaderboard");
	list.replaceChildren();
	const entries = state.leaderboard.filter(
		(entry) => recordSuite(entry) === state.resultSuite,
	);
	if (!entries.length)
		return setText(
			list,
			`No ${suiteLabel(state.resultSuite)} leaderboard results yet.`,
		);
	if (state.resultSuite === "swe_mini") {
		renderSweMiniLeaderboard(list, entries);
		return;
	}
	renderLmEvalLeaderboard(list, entries);
}

function renderLmEvalLeaderboard(list, entries) {
	const table = document.createElement("table");
	table.className = "leaderboard-table";
	const thead = document.createElement("thead");
	const header = document.createElement("tr");
	[
		"#",
		"Model",
		"Status",
		"Tasks",
		"Runtime backend",
		"Context",
		"Tok/s",
		"TTFT",
		"Overall",
		...LEADERBOARD_CATEGORIES,
	].forEach((name) => {
		const th = document.createElement("th");
		th.textContent = name;
		header.append(th);
	});
	thead.append(header);
	const tbody = document.createElement("tbody");
	entries.forEach((entry, index) => {
		const model = modelForEntry(entry);
		const modelName = entry.model || entry.model_id || "unknown model";
		const tr = document.createElement("tr");
		tr.append(
			leaderboardCell(`#${index + 1}`, "rank-cell"),
			leaderboardCell(modelName, "model-cell", modelName),
			leaderboardCell(entry.status || (entry.partial ? "partial" : "—")),
			leaderboardCell(formatTaskCoverage(entry)),
			leaderboardCell(modelBackendLabel(entry, model)),
			leaderboardCell(
				formatContext(entry.context_window || model?.context_window),
			),
			leaderboardCell(formatRate(entry.generation_tok_s)),
			leaderboardCell(formatSeconds(entry.ttft_s)),
			leaderboardCell(
				formatScore(entry.overall_score),
				"score-cell overall-score",
			),
		);
		LEADERBOARD_CATEGORIES.forEach((category) => {
			const categoryScore = categoryScoreFor(entry, category);
			const cell = leaderboardCell(
				formatScore(categoryScore?.score),
				"score-cell category-score",
			);
			if (categoryScore?.tasks?.length)
				cell.title = categoryScore.tasks.join(", ");
			tr.append(cell);
		});
		tbody.append(tr);
	});
	table.append(thead, tbody);
	list.append(table);
}

function formatTaskCoverage(entry) {
	const completed = Number(entry.result_task_count);
	const requested = Number(entry.requested_task_count);
	if (Number.isFinite(completed) && Number.isFinite(requested) && requested > 0)
		return `${completed}/${requested}`;
	if (Number.isFinite(completed)) return String(completed);
	return "—";
}

function renderSweMiniLeaderboard(list, entries) {
	const table = document.createElement("table");
	table.className = "leaderboard-table";
	const thead = document.createElement("thead");
	const header = document.createElement("tr");
	[
		"#",
		"Model",
		"Runtime backend",
		"Judge",
		"Passed",
		"Success",
		"Avg duration",
	].forEach((name) => {
		const th = document.createElement("th");
		th.textContent = name;
		header.append(th);
	});
	thead.append(header);
	const tbody = document.createElement("tbody");
	entries.forEach((entry, index) => {
		const model = modelForEntry(entry);
		const modelName = entry.model || entry.model_id || "unknown model";
		const tr = document.createElement("tr");
		tr.append(
			leaderboardCell(`#${index + 1}`, "rank-cell"),
			leaderboardCell(modelName, "model-cell", modelName),
			leaderboardCell(modelBackendLabel(entry, model)),
			leaderboardCell(displayJudgeModel(entry.judge_model)),
			leaderboardCell(`${entry.passed_tasks ?? 0}/${entry.total_tasks ?? 0}`),
			leaderboardCell(
				formatScore(entry.overall_score),
				"score-cell overall-score",
			),
			leaderboardCell(formatDurationMs(entry.average_duration_ms)),
		);
		tbody.append(tr);
	});
	table.append(thead, tbody);
	list.append(table);
}

function renderResults() {
	renderLeaderboard();
	const suiteRows = state.rows.filter(
		(row) => recordSuite(row) === state.resultSuite,
	);
	const metrics = [...new Set(suiteRows.map((row) => row.metric))].sort();
	const metricSelect = $("metricSelect");
	const previous = metricSelect.value;
	metricSelect.replaceChildren();
	metrics.forEach((metric) => {
		const option = document.createElement("option");
		option.value = metric;
		option.textContent = metric;
		metricSelect.append(option);
	});
	if (metrics.includes(previous)) metricSelect.value = previous;
	const metric = metricSelect.value || metrics[0];
	const rows = suiteRows.filter((row) => row.metric === metric);
	renderChart(rows, metric);
	renderTable(rows);
}
function renderChart(rows, metric) {
	const chart = $("chart");
	chart.replaceChildren();
	if (!rows.length) return setText(chart, "No numeric results yet.");
	const width = 1000,
		rowHeight = 42,
		height = Math.max(240, rows.length * rowHeight + 50);
	const maxValue = Math.max(...rows.map((row) => Math.abs(row.value)), 1);
	const svg = document.createElementNS(SVG_NS, "svg");
	svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
	svg.setAttribute("role", "img");
	svg.setAttribute("aria-label", `${metric || "selected metric"} chart`);
	rows.forEach((row, index) => {
		const y = 30 + index * rowHeight;
		const barWidth = Math.max(2, (Math.abs(row.value) / maxValue) * 650);
		svg.append(
			svgText(10, y + 16, `${row.model} · ${row.task}`, "bar-label"),
			svgRect(290, y, barWidth, 24),
			svgText(300 + barWidth, y + 16, formatValue(row.value), "axis-label"),
		);
	});
	chart.append(svg);
}
function renderTable(rows) {
	const wrap = $("resultTable");
	wrap.replaceChildren();
	if (!rows.length) return;
	const table = document.createElement("table");
	const thead = document.createElement("thead");
	const header = document.createElement("tr");
	["Model", "Task", "Metric", "Value", "Samples", "Job"].forEach((name) => {
		const th = document.createElement("th");
		th.textContent = name;
		header.append(th);
	});
	thead.append(header);
	const tbody = document.createElement("tbody");
	rows.forEach((row) => {
		const tr = document.createElement("tr");
		[
			row.model,
			row.task,
			row.metric,
			formatValue(row.value),
			row.samples ?? "",
			row.job_id,
		].forEach((value) => {
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
	const jobIds = state.jobs
		.filter(
			(job) =>
				state.selectedJobs.has(job.id) && ACTIVE_JOB_STATUSES.has(job.status),
		)
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
		await loadSelectedJobLog();
	} catch (error) {
		$("setupMessage").textContent = error.message;
	}
}

async function clearSelectedJobs() {
	const jobIds = [...state.selectedJobs];
	if (!jobIds.length) return;
	try {
		const payload = await api("/api/jobs/clear", {
			method: "POST",
			body: JSON.stringify({ job_ids: jobIds }),
		});
		state.jobs = payload.jobs || [];
		state.selectedJobs.clear();
		state.selectedJobId = null;
		state.jobDetails.clear();
		state.jobCommands.clear();
		$("jobLog").textContent = "";
		$("setupMessage").textContent =
			`Cleared ${payload.cleared} selected job(s).`;
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
		state.selectedJobId = null;
		state.jobDetails.clear();
		state.jobCommands.clear();
		$("jobLog").textContent = "";
		$("setupMessage").textContent = `Cleared ${payload.cleared} failed job(s).`;
		renderJobs();
		invalidateResultRows();
		await loadResults({ forceRows: true });
	} catch (error) {
		$("setupMessage").textContent = error.message;
	}
}
async function rerunSelectedJobs() {
	const jobIds = [...state.selectedJobs];
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
		state.selectedJobs.clear();
		if (created.length) state.selectedJobId = created.at(-1).id;
		$("setupMessage").textContent = `Started ${created.length} rerun job(s).`;
		await loadJobs({ refreshResultsOnTransition: false });
		invalidateResultRows();
		await loadResults({ forceRows: true });
		await loadSelectedJobLog({ forceScroll: true });
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
			`Select at least one model and one ${suiteLabel(suite)} task.`);
	const body = {
		suite,
		model_ids: modelIds,
		tasks,
		openai_base_url: $("openaiBaseUrl").value.trim(),
		llamacpp_backend: $("llamacppBackend").value || null,
		max_concurrent_jobs: Number($("maxConcurrentJobs").value || 1),
	};
	if (suite === "swe_mini") {
		Object.assign(body, {
			judge_model: $("sweJudgeModel").value.trim() || DEFAULT_SWE_JUDGE_MODEL,
			swe_timeout: Number($("sweTimeout").value || 30),
			pass_count: Number($("swePassCount").value || 1),
			context_window: numberOrNull($("sweContextWindow").value),
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
	badge.className = "badge progress";
	badge.textContent = text;
	return badge;
}
function progressText(job) {
	const progress = job.progress;
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
function suiteBadge(job) {
	const badge = document.createElement("span");
	badge.className = "badge suite";
	badge.textContent = suiteLabel(jobSuite(job));
	return badge;
}
function jobDetailMeta(job) {
	const details = div("job-meta");
	const options = job.swe_options || {};
	const evalOptions = job.eval_options || {};
	const batchProgress = job.batch_progress || {};
	const values = [
		`Suite: ${suiteLabel(jobSuite(job))}`,
		progressText(job) ? `Progress: ${progressText(job)}` : null,
		job.rerun_of ? `Rerun of: ${job.rerun_of}` : null,
		evalOptions.task_batch_size
			? `Task batch size: ${evalOptions.task_batch_size}`
			: null,
		batchProgress.total
			? `Batches: ${batchProgress.completed || 0}/${batchProgress.total}`
			: null,
		options.judge_model
			? `Judge: ${displayJudgeModel(options.judge_model)}`
			: null,
		options.pass_count ? `Pass attempts: ${options.pass_count}` : null,
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
		model.context_window
			? `${model.context_window.toLocaleString()} ctx`
			: null,
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
function formatRate(value) {
	return value === null || value === undefined || Number.isNaN(Number(value))
		? "—"
		: `${Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 })}`;
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
	const isSweMini = state.activeSuite === "swe_mini";
	$("taskPanelTitle").textContent = `${suiteLabel(state.activeSuite)} tasks`;
	$("taskFilter").placeholder = isSweMini
		? "Type to search SWE Mini tasks or repos"
		: "Type to search 14k+ tasks";
	$("taskViewModeControl").hidden = isSweMini;
	$("lmEvalCategoryFilters").hidden = isSweMini;
	$("lmEvalCompatibilityFilters").hidden = isSweMini;
	$("lmEvalBenchmarkOptions").hidden = isSweMini;
	$("sweMiniBenchmarkOptions").hidden = !isSweMini;
	$("sweMiniJudgeHint").hidden = !isSweMini;
	$("taskHint").textContent = isSweMini
		? "SWE Mini tasks run in Docker SWE-bench containers and are judged by the selected judge model."
		: "OpenAI-compatible chat backends are generation oriented. Use generate_until tasks first.";
	for (const button of [$("suiteLmEval"), $("suiteSweMini")]) {
		button.classList.toggle(
			"active",
			button.dataset.suite === state.activeSuite,
		);
	}
	for (const button of [$("leaderboardLmEval"), $("leaderboardSweMini")]) {
		button.classList.toggle(
			"active",
			button.dataset.suite === state.resultSuite,
		);
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
$("selectAllJobs").addEventListener("change", toggleAllJobs);
$("cancelSelectedJobs").addEventListener("click", cancelSelectedJobs);
$("clearSelectedJobs").addEventListener("click", clearSelectedJobs);
$("rerunSelectedJobs").addEventListener("click", rerunSelectedJobs);
$("clearFailedJobs").addEventListener("click", clearFailedJobs);
$("refreshJobs").addEventListener("click", () =>
	Promise.all([
		loadJobs(),
		loadResults({ forceRows: true }),
		loadSelectedJobLog(),
	]),
);
$("refreshAll").addEventListener("click", () =>
	Promise.all([
		loadModels(),
		loadTasks(),
		loadJobs(),
		loadResults({ forceRows: true }),
		loadSelectedJobLog(),
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
$("metricSelect").addEventListener("change", renderResults);
$("suiteLmEval").addEventListener("click", () =>
	selectBenchmarkSuite("lm_eval"),
);
$("suiteSweMini").addEventListener("click", () =>
	selectBenchmarkSuite("swe_mini"),
);
$("leaderboardLmEval").addEventListener("click", () =>
	void selectResultSuite("lm_eval"),
);
$("leaderboardSweMini").addEventListener("click", () =>
	void selectResultSuite("swe_mini"),
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
		await loadJobs();
		jobPollFailures = 0;
		const selectedJob = state.jobs.find(
			(job) => job.id === state.selectedJobId,
		);
		if (selectedJob && ACTIVE_JOB_STATUSES.has(selectedJob.status)) {
			await loadSelectedJobLog();
		}
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
	scheduleJobPoll();
	void loadResults();
}

void bootstrap();

const state = {
	models: [],
	tasks: [],
	jobs: [],
	rows: [],
	leaderboard: [],
	selectedJobId: null,
	selectedJobs: new Set(),
	expandedJobs: new Set(),
	selectedModels: new Set(),
	selectedTasks: new Set(),
	visibleTaskNames: [],
	hasAutoSelectedTask: false,
	taskPage: 0,
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

async function api(path, options = {}) {
	const response = await fetch(path, {
		headers: { "Content-Type": "application/json", ...(options.headers || {}) },
		...options,
	});
	const payload = await response.json();
	if (!response.ok) throw new Error(payload.error || response.statusText);
	return payload;
}

async function loadModels() {
	const base = encodeURIComponent($("openaiBaseUrl").value.trim());
	setText($("modelList"), "Loading models…");
	try {
		const payload = await api(`/api/models?base_url=${base}`);
		state.models = payload.models || [];
		renderModels();
		renderResults();
	} catch (error) {
		setText($("modelList"), `Could not load models: ${error.message}`);
	}
}
async function loadTasks() {
	state.visibleTaskNames = [];
	setTaskLoading(true);
	$("selectVisibleTasks").disabled = true;
	$("unselectVisibleTasks").disabled = true;
	setText($("taskList"), "Loading lm-eval tasks…");
	try {
		const payload = await api("/api/tasks");
		state.tasks = payload.tasks || [];
		renderTasks();
	} catch (error) {
		setText($("taskList"), `Could not load tasks: ${error.message}`);
	} finally {
		setTaskLoading(false);
	}
}
async function loadJobs() {
	const payload = await api("/api/jobs");
	state.jobs = payload.jobs || [];
	renderJobs();
}
async function loadResults() {
	try {
		const payload = await api("/api/results");
		state.rows = payload.rows || [];
		state.leaderboard = payload.leaderboard || [];
		renderResults();
	} catch (error) {
		setText($("leaderboard"), `Could not load results: ${error.message}`);
		setText($("chart"), `Could not load results: ${error.message}`);
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

function renderTasks() {
	const list = $("taskList");
	list.replaceChildren();
	if (
		!state.selectedTasks.size &&
		state.tasks.length &&
		!state.hasAutoSelectedTask
	) {
		state.selectedTasks.add(state.tasks[0].name);
		state.hasAutoSelectedTask = true;
	}
	renderSelectedTasks();
	const filter = $("taskFilter").value.trim().toLowerCase();
	const hideIncompatible = $("hideIncompatibleTasks").checked;
	const hideGated = $("hideGatedTasks").checked;
	const hideNonEnglish = $("hideNonEnglishTasks").checked;
	const selectedCategories = selectedTaskCategories();
	const matchingTasks = state.tasks.filter((task) => {
		if (hideIncompatible && task.compatibility === "incompatible") return false;
		if (hideGated && task.compatibility === "gated") return false;
		if (hideNonEnglish && task.language_scope === "non_english") return false;
		if (!selectedCategories.has(task.category || "Other")) return false;
		return `${task.name} ${task.description || ""} ${task.compatibility || ""} ${task.category || ""}`
			.toLowerCase()
			.includes(filter);
	});
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
		checkbox.addEventListener("change", () => {
			checkbox.checked
				? state.selectedJobs.add(job.id)
				: state.selectedJobs.delete(job.id);
			renderSelectedJobs();
		});
		const details = document.createElement("details");
		details.className = "job-details";
		details.open = state.expandedJobs.has(job.id);
		details.addEventListener("toggle", () => {
			details.open
				? state.expandedJobs.add(job.id)
				: state.expandedJobs.delete(job.id);
		});
		const summary = document.createElement("summary");
		summary.className = "job-summary";
		summary.append(
			summaryBlock(job.model_id, `Job ${job.id}`),
			statusBadge(job),
		);
		summary.addEventListener("click", () => selectJob(job.id));
		const expanded = div("job-expanded");
		const header = div("job-expanded-header");
		header.append(
			summaryBlock(job.model_id, `Job ${job.id}`),
			statusBadge(job),
		);
		const taskList = document.createElement("ul");
		taskList.className = "job-task-list";
		job.tasks.forEach((taskName) => {
			const taskItem = document.createElement("li");
			taskItem.textContent = taskName;
			taskList.append(taskItem);
		});
		expanded.append(header, taskList);
		details.append(summary, expanded);
		row.append(checkbox, details);
		list.append(row);
	});
	renderSelectedJobs();
	if (!state.selectedJobId && state.jobs.length)
		selectJob(state.jobs[state.jobs.length - 1].id);
}
function renderSelectedJobs() {
	const count = state.selectedJobs.size;
	$("selectedJobCount").textContent = `${count.toLocaleString()} selected`;
	$("clearSelectedJobs").disabled = count === 0;
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
	if (!state.selectedJobId) return;
	const log = $("jobLog");
	const autoScroll = forceScroll || shouldAutoScrollLog(log);
	try {
		const { job } = await api(`/api/jobs/${state.selectedJobId}`);
		const content = `$ ${job.command.join(" ")}\n\n${job.log_tail || "No log output yet."}`;
		if (log.textContent !== content) log.textContent = content;
	} catch (error) {
		log.textContent = error.message;
	}
	if (autoScroll) scrollLogToBottom(log);
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
	if (!state.leaderboard.length)
		return setText(list, "No leaderboard results yet.");
	const table = document.createElement("table");
	table.className = "leaderboard-table";
	const thead = document.createElement("thead");
	const header = document.createElement("tr");
	[
		"#",
		"Model",
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
	state.leaderboard.forEach((entry, index) => {
		const model = modelForEntry(entry);
		const tr = document.createElement("tr");
		tr.append(
			leaderboardCell(`#${index + 1}`, "rank-cell"),
			leaderboardCell(
				entry.model || entry.model_id || "unknown model",
				"model-cell",
			),
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

function renderResults() {
	renderLeaderboard();
	const metrics = [...new Set(state.rows.map((row) => row.metric))].sort();
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
	const rows = state.rows.filter((row) => row.metric === metric);
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
		$("jobLog").textContent = "";
		$("setupMessage").textContent =
			`Cleared ${payload.cleared} selected job(s).`;
		renderJobs();
		await loadResults();
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
		$("jobLog").textContent = "";
		$("setupMessage").textContent = `Cleared ${payload.cleared} failed job(s).`;
		renderJobs();
		await loadResults();
	} catch (error) {
		$("setupMessage").textContent = error.message;
	}
}
async function startJobs() {
	const modelIds = [...state.selectedModels],
		tasks = [...state.selectedTasks];
	if (!modelIds.length || !tasks.length)
		return ($("setupMessage").textContent =
			"Select at least one model and one task.");
	const body = {
		model_ids: modelIds,
		tasks,
		openai_base_url: $("openaiBaseUrl").value.trim(),
		llamacpp_backend: $("llamacppBackend").value || null,
		limit: $("limit").value.trim() || null,
		num_fewshot:
			$("numFewshot").value === "" ? null : Number($("numFewshot").value),
		max_gen_toks: Number($("maxGenToks").value),
		timeout: Number($("timeout").value),
		num_concurrent: Number($("numConcurrent").value),
		max_concurrent_jobs: Number($("maxConcurrentJobs").value || 1),
		batch_size: $("batchSize").value.trim() || "1",
		apply_chat_template: $("applyChatTemplate").checked,
		fewshot_as_multiturn: $("fewshotAsMultiturn").checked,
		log_samples: $("logSamples").checked,
	};
	$("setupMessage").textContent = "Starting…";
	try {
		const payload = await api("/api/jobs", {
			method: "POST",
			body: JSON.stringify(body),
		});
		$("setupMessage").textContent = `Started ${payload.jobs.length} job(s).`;
		await loadJobs();
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
function specificRuntimeBackend(value) {
	return value && value !== "llamacpp" ? value : null;
}
function modelBackendLabel(entry, model) {
	return (
		specificRuntimeBackend(entry.provider_backend) ||
		specificRuntimeBackend(entry.lemonade_backend) ||
		specificRuntimeBackend(model?.runtime_backend) ||
		specificRuntimeBackend(model?.recipe) ||
		"—"
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
function leaderboardCell(value, className = "") {
	const cell = document.createElement("td");
	if (className) cell.className = className;
	cell.textContent = value ?? "—";
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

$("refreshModels").addEventListener("click", loadModels);
$("modelFilter").addEventListener("input", renderModels);
$("selectAllJobs").addEventListener("change", toggleAllJobs);
$("clearSelectedJobs").addEventListener("click", clearSelectedJobs);
$("clearFailedJobs").addEventListener("click", clearFailedJobs);
$("refreshJobs").addEventListener("click", () =>
	Promise.all([loadJobs(), loadResults(), loadSelectedJobLog()]),
);
$("refreshAll").addEventListener("click", () =>
	Promise.all([
		loadModels(),
		loadTasks(),
		loadJobs(),
		loadResults(),
		loadSelectedJobLog(),
	]),
);
$("startJobs").addEventListener("click", startJobs);
$("selectVisibleTasks").addEventListener("click", selectVisibleTasks);
$("unselectVisibleTasks").addEventListener("click", unselectVisibleTasks);
$("taskFilter").addEventListener("input", resetTaskPage);
$("hideIncompatibleTasks").addEventListener("change", resetTaskPage);
$("hideGatedTasks").addEventListener("change", resetTaskPage);
$("hideNonEnglishTasks").addEventListener("change", resetTaskPage);
TASK_CATEGORY_FILTERS.forEach(({ id }) =>
	$(id).addEventListener("change", resetTaskPage),
);
$("taskPrev").addEventListener("click", () => changeTaskPage(-1));
$("taskNext").addEventListener("click", () => changeTaskPage(1));
$("metricSelect").addEventListener("change", renderResults);

Promise.all([loadModels(), loadTasks(), loadJobs(), loadResults()]);
setInterval(
	() => Promise.all([loadJobs(), loadResults(), loadSelectedJobLog()]),
	5000,
);

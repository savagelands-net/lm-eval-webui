#!/usr/bin/env bash
# Restore host ownership of files created by Docker containers.

resolve_ownership_target() {
	local result_path="$1"

	case "$result_path" in
	benchmark_results/*/*)
		local rest="${result_path#benchmark_results/}"
		local platform="${rest%%/*}"
		printf 'benchmark_results/%s\n' "$platform"
		;;
	*)
		printf '%s\n' "$result_path"
		;;
	esac
}

restore_docker_result_ownership() {
	local result_path="${1:-}"
	local chown_image="${2:-}"

	if [ -z "$result_path" ] || [ -z "$chown_image" ]; then
		return 0
	fi

	if [ "${PI_BENCH_SKIP_CHOWN:-0}" = "1" ]; then
		echo "[INFO] Skipping Docker result ownership restore via PI_BENCH_SKIP_CHOWN=1."
		return 0
	fi

	local run_dir="${PI_BENCH_RUN_DIR:-${PI_BENCH_DIR:-$(pwd)}}"
	local target
	target="$(resolve_ownership_target "$result_path")"

	if [ ! -e "$run_dir/$target" ]; then
		return 0
	fi

	local host_uid="${PI_BENCH_HOST_UID:-$(id -u)}"
	local host_gid="${PI_BENCH_HOST_GID:-$(id -g)}"

	if [ "$host_uid" = "0" ]; then
		return 0
	fi

	echo "[INFO] Restoring host ownership for $target to $host_uid:$host_gid"
	if ! docker run --rm \
		-v "$run_dir:/pi-bench:z" \
		--entrypoint chown \
		"$chown_image" \
		-R "$host_uid:$host_gid" "/pi-bench/$target"; then
		echo "[WARN] Failed to restore ownership for $target; you may need to run: sudo chown -R $host_uid:$host_gid '$run_dir/$target'" >&2
	fi
}

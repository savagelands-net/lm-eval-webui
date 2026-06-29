#!/usr/bin/env bash
# Prepare a temporary Pi auth directory for Docker containers.
#
# This keeps browser-login/OAuth tokens out of job JSON and avoids mounting the
# host ~/.pi tree directly. The caller receives PI_AUTH_DOCKER_ARGS containing
# the Docker env/mount flags for PI_CODING_AGENT_DIR.

PI_AUTH_DOCKER_ARGS=()
PI_BENCH_AUTH_DIR="${PI_BENCH_AUTH_DIR:-}"

prepare_pi_auth_for_docker() {
	PI_AUTH_DOCKER_ARGS=()

	if [ "${PI_BENCH_DISABLE_PI_AUTH:-0}" = "1" ]; then
		echo "[INFO] Pi auth mount disabled via PI_BENCH_DISABLE_PI_AUTH=1."
		return 0
	fi

	local source_auth="${PI_BENCH_AUTH_SOURCE:-${HOME:-}/.pi/agent/auth.json}"
	if [ ! -f "$source_auth" ]; then
		if [ "${PI_BENCH_REQUIRE_PI_AUTH:-0}" = "1" ]; then
			echo "[ERROR] Pi auth file not found: $source_auth" >&2
			echo "[ERROR] Run 'pi' and login for openai-codex, or set PI_BENCH_AUTH_SOURCE." >&2
			return 1
		fi
		echo "[INFO] Pi auth file not found at $source_auth; continuing without PI_CODING_AGENT_DIR mount."
		return 0
	fi

	local tmp_parent="${PI_BENCH_AUTH_TMP_PARENT:-/tmp}"
	local auth_dir
	auth_dir="$(mktemp -d "$tmp_parent/lm-eval-pi-auth.XXXXXX")"
	chmod 700 "$auth_dir"
	cp "$source_auth" "$auth_dir/auth.json"
	chmod 600 "$auth_dir/auth.json"

	PI_BENCH_AUTH_SOURCE="$source_auth"
	PI_BENCH_AUTH_DIR="$auth_dir"
	export PI_BENCH_AUTH_SOURCE PI_BENCH_AUTH_DIR

	local container_dir="${PI_BENCH_AUTH_CONTAINER_DIR:-/pi-auth}"
	# shellcheck disable=SC2034 # Consumed by scripts that source this helper.
	PI_AUTH_DOCKER_ARGS=(
		-e "PI_CODING_AGENT_DIR=$container_dir"
		-v "$auth_dir:$container_dir:z"
	)

	echo "[INFO] Copied Pi auth to temporary directory: $auth_dir"
	echo "[INFO] Docker containers will use PI_CODING_AGENT_DIR=$container_dir"
}

cleanup_pi_auth_for_docker() {
	if [ -z "${PI_BENCH_AUTH_DIR:-}" ] || [ ! -d "$PI_BENCH_AUTH_DIR" ]; then
		return 0
	fi

	if [ "${PI_BENCH_KEEP_PI_AUTH_COPY:-0}" = "1" ]; then
		echo "[INFO] Keeping temporary Pi auth directory: $PI_BENCH_AUTH_DIR"
		return 0
	fi

	case "$PI_BENCH_AUTH_DIR" in
	/tmp/lm-eval-pi-auth.* | */lm-eval-pi-auth.*)
		rm -rf "$PI_BENCH_AUTH_DIR"
		;;
	*)
		echo "[WARN] Refusing to remove unexpected PI_BENCH_AUTH_DIR: $PI_BENCH_AUTH_DIR" >&2
		;;
	esac
}

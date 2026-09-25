#!/bin/sh
# EDM-ARS installer for macOS and Linux.
#
# Quick route (downloads and runs this script):
#   curl -LsSf https://github.com/cgpan/edm-ars-public/releases/latest/download/install.sh | sh
#
# Careful route (read it first; see install/README.md):
#   curl -LsSfO https://github.com/cgpan/edm-ars-public/releases/latest/download/install.sh
#   less install.sh
#   sh install.sh
#
# Options (after `sh -s --` when piping, e.g. `... | sh -s -- --yes`):
#   --yes              do not pause to confirm the plan
#   --no-onboard       do not start `edmars setup` at the end
#   --dir DIR          install into DIR instead of the default folder
#   --version X.Y.Z    install this release instead of the latest one
#   --from-local PATH  install from a local checkout or edm-ars-X.Y.Z.tar.gz
#                      (for testing; also EDMARS_INSTALL_SOURCE=PATH)
#   --bin-dir DIR      put the `edmars` command in DIR (default ~/.local/bin)
#   --no-modify-path   do not add the command's folder to your PATH
#   --dry-run          show what would happen and change nothing
#   --help             show this help
#
# Everything is installed inside your user account; no administrator
# rights are needed and nothing system-wide changes. What goes where:
#   <dir>/app/<version>     EDM-ARS itself (the previous version is kept)
#   <dir>/venv-<version>    its private Python packages
#   <dir>/python            a private Python 3.11 (your own Python is untouched)
#   <dir>/uv                the uv tool, only if you do not have it already
#   ~/.local/bin/edmars     the command you type
# Default <dir>: ~/Library/Application Support/edm-ars (macOS) or
# ~/.local/share/edm-ars (Linux). To remove it: run `edmars uninstall`
# (settings and keys), then delete <dir>, the command and the PATH block
# this script added; <dir>/install.json lists everything it created.
#
# The whole script runs from main() at the very end, so a download that
# is cut off halfway cannot run half an installer.

set -eu

EDMARS_REPO="cgpan/edm-ars-public"
# uv installs Python and the packages. Pinned: the installer script at this
# URL is fixed for a given uv version, and its SHA-256 is checked below.
UV_VERSION="0.10.6"
UV_INSTALLER_SHA256="df1fea3791f6e8ad5247e2169c5edd2dc79194c1bd4af7942d68624ab3d7d1ea"
# Oldest uv we reuse if you already have one (needs `python install --no-bin`).
UV_MIN_VERSION="0.8.0"
PYTHON_SERIES="3.11"
MIN_FREE_GB=6
RAM_ADVISED_GB=16
LAUNCHER_MARK="EDM-ARS command, written by the EDM-ARS installer"

say() { printf '%s\n' "$*"; }
step() { printf '\n[%s] %s\n' "$1" "$2"; }
warn() { printf '  ! %s\n' "$*"; }
die() {
    printf '\nThe installation stopped: %s\n' "$*" >&2
    exit 1
}
have() { command -v "$1" >/dev/null 2>&1; }

# Quote a string for a POSIX shell script: 'it'\''s'.
shell_quote() {
    printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

# Escape a string for a JSON value (paths only: backslash and double quote).
json_str() {
    printf '"%s"' "$(printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')"
}

# version_ge A B: true when version A >= B (first three numeric fields).
version_ge() {
    awk -v a="$1" -v b="$2" 'BEGIN {
        split(a, x, "."); split(b, y, ".")
        for (i = 1; i <= 3; i++) {
            if ((x[i] + 0) > (y[i] + 0)) exit 0
            if ((x[i] + 0) < (y[i] + 0)) exit 1
        }
        exit 0
    }'
}

sha256_of() {
    if have sha256sum; then
        sha256sum "$1" | cut -d ' ' -f 1
    elif have shasum; then
        shasum -a 256 "$1" | cut -d ' ' -f 1
    elif have openssl; then
        openssl dgst -sha256 "$1" | sed 's/^.*= *//'
    else
        die "no SHA-256 tool found (sha256sum, shasum or openssl)."
    fi
}

# fetch URL DEST. A file:// URL or a plain folder path is copied, which
# lets EDMARS_RELEASE_BASE_URL point at a local dist/ folder for testing.
fetch() {
    case "$1" in
        file://*)
            cp "${1#file://}" "$2"
            ;;
        http://* | https://*)
            if have curl; then
                curl --proto '=https' --tlsv1.2 -fsSL --retry 3 -o "$2" "$1"
            elif have wget; then
                wget -q --https-only -O "$2" "$1"
            else
                die "neither curl nor wget is installed; install one of them and try again."
            fi
            ;;
        *)
            cp "$1" "$2"
            ;;
    esac
}

# Nearest folder that already exists, for disk-space checks.
existing_parent() {
    ep_path=$1
    while [ ! -d "$ep_path" ]; do
        ep_next=$(dirname "$ep_path")
        [ "$ep_next" = "$ep_path" ] && break
        ep_path=$ep_next
    done
    printf '%s' "$ep_path"
}

# Name of the cloud-sync service whose folder contains PATH, if any.
sync_provider() {
    case "/$1/" in
        */OneDrive/* | */OneDrive\ -\ */*) say "OneDrive" ;;
        */Dropbox/* | */Dropbox\ \(*) say "Dropbox" ;;
        *"/Google Drive/"* | */GoogleDrive*/* | *"/My Drive/"*) say "Google Drive" ;;
        *"/Mobile Documents/"* | *"/iCloud Drive/"*) say "iCloud Drive" ;;
        */Library/CloudStorage/*) say "a cloud-storage service" ;;
        *) return 1 ;;
    esac
}

usage() {
    cat <<'EOF'
EDM-ARS installer for macOS and Linux.

Usage: sh install.sh [options]
   or: curl -LsSf https://github.com/cgpan/edm-ars-public/releases/latest/download/install.sh | sh -s -- [options]

Options:
  --yes              do not pause to confirm the plan
  --no-onboard       do not start `edmars setup` at the end
  --dir DIR          install into DIR instead of the default folder
  --version X.Y.Z    install this release instead of the latest one
  --from-local PATH  install from a local checkout or edm-ars-X.Y.Z.tar.gz
                     (for testing; also EDMARS_INSTALL_SOURCE=PATH)
  --bin-dir DIR      put the `edmars` command in DIR (default ~/.local/bin)
  --no-modify-path   do not add the command's folder to your PATH
  --dry-run          show what would happen and change nothing
  --help             show this help

Everything is installed inside your user account; no administrator rights
are needed. To remove it: run `edmars uninstall` (settings and keys), then
delete the install folder and the command; install.json in the install
folder lists everything this installer created.
EOF
}

# Append the PATH block to one shell start-up file (once).
add_path_block() {
    if grep -qs '>>> edm-ars >>>' "$1"; then
        return 0
    fi
    # The block is written for the shell that later reads the file, so the
    # $PATH (and $HOME) in it must stay unexpanded here.
    # shellcheck disable=SC2016
    {
        printf '\n# >>> edm-ars >>> (added by the EDM-ARS installer; delete this block to undo)\n'
        printf 'case ":${PATH}:" in *":%s:"*) ;; *) export PATH="%s:$PATH" ;; esac\n' "$PATH_EXPR" "$PATH_EXPR"
        printf '# <<< edm-ars <<<\n'
    } >>"$1"
    PATH_JSON="$PATH_JSON${PATH_JSON:+, }$(json_str "$1")"
    PATH_LIST="$PATH_LIST
      $1"
}

# --------------------------------------------------------------------------
main() {
    YES=0
    NO_ONBOARD=0
    DRY_RUN=0
    NO_MODIFY_PATH=0
    APP_BASE=${EDMARS_INSTALL_DIR:-}
    DIR_EXPLICIT=0
    [ -n "$APP_BASE" ] && DIR_EXPLICIT=1
    REQ_VERSION=""
    FROM_LOCAL=${EDMARS_INSTALL_SOURCE:-}
    BIN_DIR=${EDMARS_BIN_DIR:-}

    while [ $# -gt 0 ]; do
        case "$1" in
            -y | --yes) YES=1 ;;
            --no-onboard) NO_ONBOARD=1 ;;
            --no-modify-path) NO_MODIFY_PATH=1 ;;
            --dry-run) DRY_RUN=1 ;;
            -h | --help) usage; exit 0 ;;
            --dir | --version | --from-local | --bin-dir)
                [ $# -ge 2 ] || die "$1 needs a value (see --help)."
                case "$1" in
                    --dir) APP_BASE=$2; DIR_EXPLICIT=1 ;;
                    --version) REQ_VERSION=$2 ;;
                    --from-local) FROM_LOCAL=$2 ;;
                    --bin-dir) BIN_DIR=$2 ;;
                esac
                shift
                ;;
            --dir=*) APP_BASE=${1#*=}; DIR_EXPLICIT=1 ;;
            --version=*) REQ_VERSION=${1#*=} ;;
            --from-local=*) FROM_LOCAL=${1#*=} ;;
            --bin-dir=*) BIN_DIR=${1#*=} ;;
            *) die "unknown option '$1' (see --help)." ;;
        esac
        shift
    done

    [ -n "${HOME:-}" ] || die "HOME is not set."
    # Python variables from the calling shell must not leak into the
    # private interpreter we are about to build and test.
    unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX 2>/dev/null || true

    HAVE_TTY=0
    if (: </dev/tty) 2>/dev/null; then HAVE_TTY=1; fi

    # ---- computer check ---------------------------------------------------
    OS=$(uname -s)
    MACHINE=$(uname -m)
    PY_REQUEST=$PYTHON_SERIES
    PLATFORM_NOTE=""
    SUPPORTED=1
    case "$OS" in
        Darwin)
            PLATFORM="macOS"
            if [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ]; then
                PLATFORM="macOS (Apple Silicon)"
                # Ask for the native build explicitly: a Terminal running
                # under Rosetta reports x86_64 and would otherwise get an
                # Intel Python, for which several scientific packages
                # publish no builds.
                PY_REQUEST="cpython-${PYTHON_SERIES}-macos-aarch64-none"
                if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ]; then
                    PLATFORM_NOTE="This Terminal is running under Rosetta; EDM-ARS will still use the native Apple Silicon Python."
                fi
            else
                PLATFORM="macOS (Intel)"
                SUPPORTED=0
                PLATFORM_NOTE="Intel Macs are not supported: numba, llvmlite and scs no longer publish Intel Mac builds of the versions EDM-ARS is tested with."
            fi
            DEFAULT_BASE="$HOME/Library/Application Support/edm-ars"
            ;;
        Linux)
            PLATFORM="Linux"
            case "$MACHINE" in
                x86_64 | amd64) PLATFORM="Linux (x86_64)" ;;
                aarch64 | arm64) PLATFORM="Linux (arm64)" ;;
                *)
                    SUPPORTED=0
                    PLATFORM_NOTE="The processor type '$MACHINE' is not supported (x86_64 or arm64 only)."
                    ;;
            esac
            if have ldd && ldd --version 2>&1 | grep -qi musl; then
                SUPPORTED=0
                PLATFORM_NOTE="musl-based Linux (for example Alpine) is not supported; use a glibc distribution such as Ubuntu, Debian or Fedora."
            fi
            DEFAULT_BASE="${XDG_DATA_HOME:-$HOME/.local/share}/edm-ars"
            ;;
        MINGW* | MSYS* | CYGWIN*)
            PLATFORM="Windows ($OS)"
            SUPPORTED=0
            PLATFORM_NOTE="On Windows, use install.ps1 in PowerShell instead of this script."
            DEFAULT_BASE="$HOME/.local/share/edm-ars"
            ;;
        *)
            PLATFORM=$OS
            SUPPORTED=0
            PLATFORM_NOTE="This system ($OS) is not supported; EDM-ARS supports macOS and Linux here, Windows via install.ps1."
            DEFAULT_BASE="$HOME/.local/share/edm-ars"
            ;;
    esac
    if [ "$SUPPORTED" = 0 ] && [ "$DRY_RUN" = 0 ]; then
        die "$PLATFORM_NOTE"
    fi

    [ -n "$APP_BASE" ] || APP_BASE=$DEFAULT_BASE
    [ -n "$BIN_DIR" ] || BIN_DIR="$HOME/.local/bin"
    case "$APP_BASE" in "~"/*) APP_BASE="$HOME/${APP_BASE#"~/"}" ;; esac
    case "$BIN_DIR" in "~"/*) BIN_DIR="$HOME/${BIN_DIR#"~/"}" ;; esac
    case "$APP_BASE" in /*) ;; *) APP_BASE="$(pwd)/$APP_BASE" ;; esac
    case "$BIN_DIR" in /*) ;; *) BIN_DIR="$(pwd)/$BIN_DIR" ;; esac
    APP_BASE=${APP_BASE%/}
    BIN_DIR=${BIN_DIR%/}
    case "$BIN_DIR" in
        *'"'* | *'$'* | *'`'* | *\\*)
            die "the command folder '$BIN_DIR' contains a quote, \$, backquote or backslash; choose another with --bin-dir."
            ;;
    esac

    SYNC=""
    if SYNC=$(sync_provider "$APP_BASE"); then
        if [ "$DIR_EXPLICIT" = 1 ] || [ "$DRY_RUN" = 1 ]; then
            warn "$APP_BASE is inside a $SYNC folder. Syncing thousands of package files is slow and can corrupt the install; a local folder is strongly recommended."
        else
            die "the install folder $APP_BASE is inside a $SYNC folder. Choose a local folder with --dir."
        fi
    else
        SYNC=""
    fi

    FREE_PARENT=$(existing_parent "$APP_BASE")
    FREE_KB=$(df -Pk "$FREE_PARENT" 2>/dev/null | awk 'NR == 2 {print $4}') || FREE_KB=""
    FREE_GB=""
    case "$FREE_KB" in
        '' | *[!0-9]*) FREE_GB="" ;;
        *) FREE_GB=$((FREE_KB / 1048576)) ;;
    esac

    RAM_GB=""
    if [ -r /proc/meminfo ]; then
        RAM_GB=$(awk '/^MemTotal:/ {printf "%d", ($2 / 1048576) + 0.5}' /proc/meminfo)
    elif have sysctl; then
        RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "")
        case "$RAM_BYTES" in
            '' | *[!0-9]*) ;;
            *) RAM_GB=$(awk -v b="$RAM_BYTES" 'BEGIN {printf "%d", (b / 1073741824) + 0.5}') ;;
        esac
    fi

    # ---- where the code comes from -----------------------------------------
    SOURCE_KIND="github-release"
    VERSION=""
    if [ -n "$FROM_LOCAL" ]; then
        case "$FROM_LOCAL" in "~"/*) FROM_LOCAL="$HOME/${FROM_LOCAL#"~/"}" ;; esac
        case "$FROM_LOCAL" in /*) ;; *) FROM_LOCAL="$(pwd)/$FROM_LOCAL" ;; esac
        if [ -d "$FROM_LOCAL" ]; then
            SOURCE_KIND="local-folder"
            [ -f "$FROM_LOCAL/src/main.py" ] || die "$FROM_LOCAL does not look like an EDM-ARS checkout (no src/main.py)."
            VERSION=$REQ_VERSION
            if [ -z "$VERSION" ] && [ -f "$FROM_LOCAL/edmars/__init__.py" ]; then
                VERSION=$(sed -n "s/^__version__[ ]*=[ ]*[\"']\([^\"']*\)[\"'].*/\1/p" "$FROM_LOCAL/edmars/__init__.py" | head -n 1)
            fi
            [ -n "$VERSION" ] || VERSION="local"
        elif [ -f "$FROM_LOCAL" ]; then
            SOURCE_KIND="local-archive"
            case "$(basename "$FROM_LOCAL")" in
                edm-ars-*.tar.gz)
                    VERSION=$(basename "$FROM_LOCAL" .tar.gz)
                    VERSION=${VERSION#edm-ars-}
                    ;;
                *) die "$FROM_LOCAL is not an edm-ars-<version>.tar.gz release archive." ;;
            esac
        else
            die "--from-local $FROM_LOCAL does not exist."
        fi
        SOURCE_TEXT="$FROM_LOCAL"
    else
        REQ_VERSION=${REQ_VERSION#v}
        [ "$REQ_VERSION" = "latest" ] && REQ_VERSION=""
        VERSION=$REQ_VERSION
        if [ -n "${EDMARS_RELEASE_BASE_URL:-}" ]; then
            RELEASE_BASE=${EDMARS_RELEASE_BASE_URL%/}
        elif [ -n "$REQ_VERSION" ]; then
            RELEASE_BASE="https://github.com/$EDMARS_REPO/releases/download/v$REQ_VERSION"
        else
            RELEASE_BASE="https://github.com/$EDMARS_REPO/releases/latest/download"
        fi
        SOURCE_TEXT="$RELEASE_BASE"
    fi
    if [ -n "$VERSION" ]; then
        case "$VERSION" in
            *[!0-9A-Za-z.+-]* | .* ) die "'$VERSION' is not a valid version." ;;
        esac
    fi

    # ---- uv ------------------------------------------------------------------
    UV=""
    UV_FOUND_VERSION=""
    UV_PRIVATE=0
    if [ -z "${EDMARS_FORCE_PRIVATE_UV:-}" ]; then
        for uv_candidate in "$(command -v uv 2>/dev/null || true)" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" "$APP_BASE/uv/uv"; do
            [ -n "$uv_candidate" ] && [ -x "$uv_candidate" ] || continue
            uv_seen=$("$uv_candidate" --version 2>/dev/null | awk '{print $2}') || continue
            [ -n "$uv_seen" ] || continue
            if version_ge "$uv_seen" "$UV_MIN_VERSION"; then
                UV=$uv_candidate
                UV_FOUND_VERSION=$uv_seen
                break
            fi
        done
    fi

    # ---- the plan --------------------------------------------------------------
    say ""
    say "EDM-ARS installer"
    say "================="
    say "Computer:      $PLATFORM${RAM_GB:+, ${RAM_GB} GB memory}${FREE_GB:+, ${FREE_GB} GB free}"
    [ -z "$PLATFORM_NOTE" ] || warn "$PLATFORM_NOTE"
    say "Install into:  $APP_BASE"
    say "Command in:    $BIN_DIR/edmars"
    if [ -n "$FROM_LOCAL" ]; then
        say "EDM-ARS from:  $SOURCE_TEXT (local copy, version label '$VERSION')"
    else
        say "EDM-ARS from:  $SOURCE_TEXT (${VERSION:-latest release})"
    fi
    say ""
    say "This will:"
    say "  1. Check this computer (system, free disk space, memory)."
    if [ -n "$UV" ]; then
        say "  2. Use the uv tool you already have ($UV, version $UV_FOUND_VERSION)."
    else
        say "  2. Download the uv tool $UV_VERSION from astral.sh into $APP_BASE/uv"
        say "     (checked against a fixed SHA-256 fingerprint; not added to your PATH)."
    fi
    say "  3. Install a private Python $PYTHON_SERIES (any Python you already have is left alone)."
    if [ -n "$FROM_LOCAL" ]; then
        say "  4. Copy EDM-ARS from $FROM_LOCAL."
    else
        say "  4. Download EDM-ARS from GitHub and check its SHA-256 fingerprint."
    fi
    say "  5. Install EDM-ARS and the packages it needs (about 1.5 GB)."
    say "  6. Create the command $BIN_DIR/edmars."
    if [ "$NO_MODIFY_PATH" = 1 ]; then
        say "  7. Leave your PATH alone (--no-modify-path)."
    else
        say "  7. Add $BIN_DIR to your PATH (your shell start-up files, your account only)."
    fi
    if [ "$NO_ONBOARD" = 1 ]; then
        say "  8. Stop there (--no-onboard); run 'edmars setup' when you are ready."
    else
        say "  8. Start the setup wizard ('edmars setup')."
    fi
    say ""
    say "No administrator rights are needed. To remove it later, run 'edmars uninstall'"
    say "(your settings and keys), then delete $APP_BASE and the command."

    if [ -n "$FREE_GB" ] && [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
        if [ "$DRY_RUN" = 1 ]; then
            warn "Only ${FREE_GB} GB free on the disk holding $FREE_PARENT; EDM-ARS needs at least ${MIN_FREE_GB} GB (packages, data and outputs)."
        else
            die "only ${FREE_GB} GB free on the disk holding $FREE_PARENT; EDM-ARS needs at least ${MIN_FREE_GB} GB (packages, data and outputs). Free some space or choose another disk with --dir."
        fi
    fi
    if [ -n "$RAM_GB" ] && [ "$RAM_GB" -lt "$RAM_ADVISED_GB" ]; then
        warn "This computer has ${RAM_GB} GB of memory. ${RAM_ADVISED_GB} GB is recommended; with less, large datasets such as HSLS:09 load slowly and can run out of memory."
    fi

    if [ "$DRY_RUN" = 1 ]; then
        say ""
        say "Dry run: nothing was downloaded or changed."
        exit 0
    fi

    if [ "$YES" = 0 ]; then
        if [ "$HAVE_TTY" = 0 ]; then
            die "there is no terminal to ask for confirmation. Re-run with --yes to install without asking."
        fi
        printf '\nContinue? [Y/n] ' >/dev/tty
        answer=""
        read -r answer </dev/tty || answer="n"
        case "$answer" in
            '' | y | Y | yes | Yes | YES) ;;
            *) say "Nothing was installed."; exit 0 ;;
        esac
    fi

    have tar || die "the 'tar' program is missing."
    have find || die "the 'find' program is missing."

    mkdir -p "$APP_BASE/app" || die "cannot create $APP_BASE."
    TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/edmars-install.XXXXXX") || die "cannot create a temporary folder."
    STAGE="$APP_BASE/app/.staging-$$"
    trap 'rm -rf "$TMP_DIR" "$STAGE"' EXIT
    trap 'exit 130' INT TERM
    # uv reads configuration from the current folder; run it from a neutral one.
    cd "$TMP_DIR"

    # ---- 2. uv -----------------------------------------------------------------
    step "2/8" "Getting uv"
    if [ -n "$UV" ]; then
        say "Using $UV ($UV_FOUND_VERSION)."
        # A uv this installer put in the install folder earlier is still ours.
        if [ "$UV" = "$APP_BASE/uv/uv" ]; then UV_PRIVATE=1; fi
    else
        fetch "https://astral.sh/uv/$UV_VERSION/install.sh" "$TMP_DIR/uv-installer.sh" \
            || die "could not download the uv installer. Check your internet connection."
        uv_sum=$(sha256_of "$TMP_DIR/uv-installer.sh")
        if [ "$uv_sum" != "$UV_INSTALLER_SHA256" ]; then
            die "the uv installer did not match its expected fingerprint, so it was not run. You can install uv yourself (https://docs.astral.sh/uv/) and run this installer again."
        fi
        # UV_UNMANAGED_INSTALL: install into that folder only, without
        # touching PATH, shell files or uv's self-update receipt.
        UV_UNMANAGED_INSTALL="$APP_BASE/uv" sh "$TMP_DIR/uv-installer.sh" \
            || die "the uv installer failed (see the messages above)."
        UV="$APP_BASE/uv/uv"
        UV_PRIVATE=1
        [ -x "$UV" ] || die "uv was not found at $UV after installing it."
    fi

    # ---- 3. Python -------------------------------------------------------------
    step "3/8" "Installing a private Python $PYTHON_SERIES"
    UV_PYTHON_INSTALL_DIR="$APP_BASE/python"
    export UV_PYTHON_INSTALL_DIR
    "$UV" python install "$PY_REQUEST" --no-bin \
        || die "uv could not install Python $PYTHON_SERIES (see the messages above)."
    BASE_PY=$("$UV" python find --managed-python --no-project "$PY_REQUEST") \
        || die "the private Python $PYTHON_SERIES was installed but cannot be found."

    # ---- 4. EDM-ARS ------------------------------------------------------------
    rm -rf "$STAGE"
    mkdir -p "$STAGE"
    if [ "$SOURCE_KIND" = "local-folder" ]; then
        step "4/8" "Copying EDM-ARS from $FROM_LOCAL"
        mkdir -p "$STAGE/src"
        # Copy the checkout without git history, caches, raw data, run
        # outputs or local secrets.
        (
            cd "$FROM_LOCAL" && find . \
                \( -name .git -o -name __pycache__ -o -name .pytest_cache \
                -o -name .mypy_cache -o -name .ruff_cache -o -name node_modules \
                -o -name .env -o -name '*.env' -o -name '*.pyc' \
                -o -path ./output -o -path ./data -o -path ./dist -o -path ./cache \
                -o -path ./.venv -o -path ./venv -o -path ./ideas -o -path ./files \
                -o -path './tmp_orch_test*' -o -path './runs/*/output*' \) -prune \
                -o \( -type f -o -type l \) -print
        ) >"$TMP_DIR/files.txt"
        (cd "$FROM_LOCAL" && tar -cf - -T "$TMP_DIR/files.txt") | (cd "$STAGE/src" && tar -xf -) \
            || die "copying $FROM_LOCAL failed."
        NEW_SRC="$STAGE/src"
    else
        if [ "$SOURCE_KIND" = "local-archive" ]; then
            step "4/8" "Checking $FROM_LOCAL"
            TARBALL=$FROM_LOCAL
            SUMS="$(dirname "$FROM_LOCAL")/SHA256SUMS"
            if [ -f "$SUMS" ]; then
                cp "$SUMS" "$TMP_DIR/SHA256SUMS"
            else
                warn "No SHA256SUMS next to $FROM_LOCAL, so its fingerprint cannot be checked."
                : >"$TMP_DIR/SHA256SUMS"
            fi
            TAR_NAME=$(basename "$FROM_LOCAL")
        else
            step "4/8" "Downloading EDM-ARS"
            fetch "$RELEASE_BASE/SHA256SUMS" "$TMP_DIR/SHA256SUMS" \
                || die "could not download $RELEASE_BASE/SHA256SUMS. Check your internet connection${REQ_VERSION:+ and that version $REQ_VERSION exists}."
            TAR_NAME=$(tr -d '\r' <"$TMP_DIR/SHA256SUMS" \
                | sed -n 's/^[0-9a-fA-F]\{64\}[ *]\{1,2\}\(edm-ars-[0-9A-Za-z.+-]*\.tar\.gz\)$/\1/p' | head -n 1)
            [ -n "$TAR_NAME" ] || die "the release has no edm-ars-<version>.tar.gz listed in SHA256SUMS."
            FOUND_VERSION=${TAR_NAME#edm-ars-}
            FOUND_VERSION=${FOUND_VERSION%.tar.gz}
            if [ -n "$REQ_VERSION" ] && [ "$FOUND_VERSION" != "$REQ_VERSION" ]; then
                die "asked for version $REQ_VERSION but the release lists $TAR_NAME."
            fi
            VERSION=$FOUND_VERSION
            case "$VERSION" in
                *[!0-9A-Za-z.+-]* | .* | '') die "'$VERSION' is not a valid version." ;;
            esac
            if [ -z "${EDMARS_RELEASE_BASE_URL:-}" ] && [ -z "$REQ_VERSION" ]; then
                # Fetch the archive from the release the checksums came from.
                TAR_URL="https://github.com/$EDMARS_REPO/releases/download/v$VERSION/$TAR_NAME"
            else
                TAR_URL="$RELEASE_BASE/$TAR_NAME"
            fi
            say "Release found: EDM-ARS $VERSION"
            TARBALL="$TMP_DIR/$TAR_NAME"
            fetch "$TAR_URL" "$TARBALL" || die "could not download $TAR_URL."
        fi
        EXPECTED=$(tr -d '\r' <"$TMP_DIR/SHA256SUMS" \
            | awk -v n="$TAR_NAME" '$2 == n || $2 == "*" n {print tolower($1); exit}')
        if [ -z "$EXPECTED" ] && [ "$SOURCE_KIND" = "local-archive" ] && [ -s "$TMP_DIR/SHA256SUMS" ]; then
            warn "The SHA256SUMS next to $FROM_LOCAL does not list $TAR_NAME, so its fingerprint cannot be checked."
        fi
        if [ -n "$EXPECTED" ]; then
            ACTUAL=$(sha256_of "$TARBALL")
            [ "$ACTUAL" = "$EXPECTED" ] \
                || die "$TAR_NAME does not match its SHA-256 fingerprint (expected $EXPECTED, got $ACTUAL). The download may be damaged or altered, so EDM-ARS was not installed."
            say "Fingerprint OK ($EXPECTED)."
        elif [ "$SOURCE_KIND" = "github-release" ]; then
            die "SHA256SUMS does not list $TAR_NAME."
        fi
        tar -xzf "$TARBALL" -C "$STAGE" || die "could not unpack $TAR_NAME."
        NEW_SRC="$STAGE/edm-ars-$VERSION"
        [ -d "$NEW_SRC" ] || die "$TAR_NAME does not contain the folder edm-ars-$VERSION."
    fi
    [ -f "$NEW_SRC/src/main.py" ] || die "the downloaded copy has no src/main.py."
    [ -f "$NEW_SRC/requirements.lock" ] || die "this copy of EDM-ARS has no requirements.lock, so its packages cannot be installed reproducibly."
    if [ ! -f "$NEW_SRC/edmars/__main__.py" ]; then
        warn "This copy has no edmars package (edmars/__main__.py); the 'edmars' command will not start."
    fi

    APP_DIR="$APP_BASE/app/$VERSION"
    VENV="$APP_BASE/venv-$VERSION"
    VENV_PY="$VENV/bin/python"
    PREVIOUS=""
    if [ -f "$APP_BASE/versions.txt" ]; then
        PREVIOUS=$(tail -n 1 "$APP_BASE/versions.txt")
    fi
    if [ -e "$APP_DIR" ]; then
        rm -rf "$APP_DIR" || die "cannot replace $APP_DIR; close any running 'edmars' and try again."
    fi
    mv "$NEW_SRC" "$APP_DIR" || die "cannot move EDM-ARS into $APP_DIR."
    rm -rf "$STAGE"

    # ---- 5. packages -----------------------------------------------------------
    step "5/8" "Installing packages (this can take several minutes)"
    rm -rf "$VENV"
    "$UV" venv --quiet --no-project --python "$BASE_PY" "$VENV" \
        || die "could not create the Python environment in $VENV."
    set -- -r "$APP_DIR/requirements.lock"
    if [ -f "$APP_DIR/requirements-cli.txt" ]; then
        set -- "$@" -r "$APP_DIR/requirements-cli.txt"
    fi
    "$UV" pip install --python "$VENV_PY" "$@" \
        || die "installing the packages failed (see the messages above). Check your internet connection and run the installer again."
    say "Checking that the main packages load..."
    if ! "$VENV_PY" -c "import numpy, pandas, scipy, sklearn, matplotlib, xgboost, shap, fitz, yaml, requests" 2>"$TMP_DIR/import.err"; then
        tail -n 5 "$TMP_DIR/import.err" >&2
        if [ "$OS" = "Darwin" ] && grep -q "libomp" "$TMP_DIR/import.err"; then
            die "XGBoost needs the OpenMP library on macOS. Install Homebrew (https://brew.sh), run 'brew install libomp', then run this installer again."
        fi
        die "the packages were installed but do not load (see the lines above)."
    fi

    # EDM-ARS itself is not a pip package: the app folder goes on the
    # environment's import path through a .pth file, so `python -m edmars`
    # (and the CLI's `import src...`) find this version's code wherever the
    # user's current folder is.
    SITE_PKGS=$("$VENV_PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
        || die "could not find the package folder of the new Python environment."
    [ -d "$SITE_PKGS" ] || die "could not find the package folder of the new Python environment."
    PTH="$SITE_PKGS/edm_ars_app.pth"
    printf '%s\n' "$APP_DIR" >"$PTH"

    # ---- 6. the command --------------------------------------------------------
    step "6/8" "Creating the edmars command"
    mkdir -p "$BIN_DIR" || die "cannot create $BIN_DIR."
    LAUNCHER="$BIN_DIR/edmars"
    if [ -e "$LAUNCHER" ] && ! grep -qF "$LAUNCHER_MARK" "$LAUNCHER" 2>/dev/null; then
        die "$LAUNCHER already exists and was not made by this installer; move it away or choose another folder with --bin-dir."
    fi
    LAUNCHER_TMP="$BIN_DIR/.edmars.$$"
    # The "$@" and backquotes belong to the launcher, not to this script.
    # shellcheck disable=SC2016
    {
        printf '#!/bin/sh\n'
        printf '# %s (EDM-ARS %s).\n' "$LAUNCHER_MARK" "$VERSION"
        printf '# Re-run the installer to update it. install.json in the install folder lists everything it created.\n'
        printf 'EDMARS_APP_ROOT=%s\n' "$(shell_quote "$APP_DIR")"
        printf 'export EDMARS_APP_ROOT\n'
        printf 'PYTHONUTF8=1\n'
        printf 'export PYTHONUTF8\n'
        printf 'unset PYTHONHOME PYTHONPATH\n'
        # uv makes environments without pip, so `edmars setup` installs
        # LSAR's packages with the uv that built this one (edmars.lsar reads
        # EDMARS_UV). That uv may not be on PATH (a private copy in
        # <dir>/uv), and a user's own uv may be removed later: then the
        # variable is left unset and edmars falls back to PATH and ensurepip
        # instead of a path that no longer exists.
        printf 'if [ -x %s ]; then\n' "$(shell_quote "$UV")"
        printf '    EDMARS_UV=%s\n' "$(shell_quote "$UV")"
        printf '    export EDMARS_UV\n'
        printf 'fi\n'
        # -P: do not put the current folder first on the import path, so
        # a user's own src/ or edmars/ folder cannot shadow ours.
        printf 'exec %s -P -m edmars "$@"\n' "$(shell_quote "$VENV_PY")"
    } >"$LAUNCHER_TMP"
    chmod 755 "$LAUNCHER_TMP"
    mv -f "$LAUNCHER_TMP" "$LAUNCHER"
    say "Created $LAUNCHER"

    # ---- 7. PATH ---------------------------------------------------------------
    PATH_JSON=""
    PATH_LIST=""
    PATH_MODIFIED=false
    step "7/8" "PATH"
    case ":$PATH:" in
        *":$BIN_DIR:"*) BIN_ON_PATH=1 ;;
        *) BIN_ON_PATH=0 ;;
    esac
    if [ "$NO_MODIFY_PATH" = 1 ]; then
        say "Left unchanged (--no-modify-path)."
    elif [ "$BIN_ON_PATH" = 1 ]; then
        say "$BIN_DIR is already on your PATH."
    else
        case "$BIN_DIR" in
            "$HOME"/*) PATH_EXPR="\$HOME/${BIN_DIR#"$HOME"/}" ;;
            *) PATH_EXPR=$BIN_DIR ;;
        esac
        add_path_block "$HOME/.profile"
        if [ -f "$HOME/.bashrc" ]; then add_path_block "$HOME/.bashrc"; fi
        if [ -f "$HOME/.bash_profile" ]; then add_path_block "$HOME/.bash_profile"; fi
        if [ -f "$HOME/.zshrc" ] || [ "${SHELL##*/}" = "zsh" ]; then
            add_path_block "$HOME/.zshrc"
        fi
        if [ -d "$HOME/.config/fish" ] || [ "${SHELL##*/}" = "fish" ]; then
            FISH_FILE="$HOME/.config/fish/conf.d/edm-ars.fish"
            mkdir -p "$HOME/.config/fish/conf.d"
            # shellcheck disable=SC2016
            {
                printf '# >>> edm-ars >>> (added by the EDM-ARS installer; delete this file to undo)\n'
                printf 'if not contains -- "%s" $PATH\n    set -gx PATH "%s" $PATH\nend\n' "$BIN_DIR" "$BIN_DIR"
                printf '# <<< edm-ars <<<\n'
            } >"$FISH_FILE"
            PATH_JSON="$PATH_JSON${PATH_JSON:+, }$(json_str "$FISH_FILE")"
            PATH_LIST="$PATH_LIST
      $FISH_FILE"
        fi
        PATH_MODIFIED=true
        say "Added $BIN_DIR to your PATH in:$PATH_LIST"
    fi

    # ---- record the install ------------------------------------------------------
    {
        if [ -f "$APP_BASE/versions.txt" ]; then
            grep -v -x -F "$VERSION" "$APP_BASE/versions.txt" || true
        fi
        printf '%s\n' "$VERSION"
    } >"$TMP_DIR/versions.txt"
    mv -f "$TMP_DIR/versions.txt" "$APP_BASE/versions.txt"
    # Keep the current and the previous version (a study started before
    # this update may still be running from it); remove older ones. Only
    # folders this installer recorded are ever removed.
    KEEP=2
    COUNT=$(wc -l <"$APP_BASE/versions.txt" | tr -d ' ')
    if [ "$COUNT" -gt "$KEEP" ]; then
        head -n $((COUNT - KEEP)) "$APP_BASE/versions.txt" | while IFS= read -r old; do
            case "$old" in
                '' | *[!0-9A-Za-z.+-]* | .*) continue ;;
            esac
            rm -rf "$APP_BASE/app/$old" "$APP_BASE/venv-$old"
        done
        tail -n "$KEEP" "$APP_BASE/versions.txt" >"$TMP_DIR/versions.txt"
        mv -f "$TMP_DIR/versions.txt" "$APP_BASE/versions.txt"
    fi
    if [ -n "$PREVIOUS" ] && [ "$PREVIOUS" != "$VERSION" ]; then
        PREVIOUS_JSON=$(json_str "$PREVIOUS")
    else
        PREVIOUS_JSON=null
    fi
    if [ "$UV_PRIVATE" = 1 ]; then UV_PRIVATE_JSON=true; else UV_PRIVATE_JSON=false; fi
    cat >"$APP_BASE/install.json" <<EOF
{
  "schema": 1,
  "installer": "install.sh",
  "version": $(json_str "$VERSION"),
  "previous_version": $PREVIOUS_JSON,
  "installed_at": $(json_str "$(date -u +%Y-%m-%dT%H:%M:%SZ)"),
  "source": $(json_str "$SOURCE_KIND"),
  "install_dir": $(json_str "$APP_BASE"),
  "app_root": $(json_str "$APP_DIR"),
  "venv": $(json_str "$VENV"),
  "python": $(json_str "$VENV_PY"),
  "pth": $(json_str "$PTH"),
  "uv": $(json_str "$UV"),
  "uv_private": $UV_PRIVATE_JSON,
  "bin_dir": $(json_str "$BIN_DIR"),
  "launcher": $(json_str "$LAUNCHER"),
  "path_modified": $PATH_MODIFIED,
  "path_files": [$PATH_JSON]
}
EOF

    # ---- 8. smoke test and setup -------------------------------------------------
    step "8/8" "Checking the edmars command"
    "$LAUNCHER" version \
        || die "the edmars command was created but did not run. Run '$LAUNCHER version' yourself to see the error."

    say ""
    say "EDM-ARS $VERSION is installed."
    RUN_NOTE=""
    if [ "$BIN_ON_PATH" = 1 ]; then
        RUN_CMD="edmars"
    elif [ "$PATH_MODIFIED" = true ]; then
        RUN_CMD="edmars"
        RUN_NOTE="   (open a new terminal window first, so it picks up the new PATH)"
    else
        RUN_CMD=$(shell_quote "$LAUNCHER")
    fi
    if [ "$NO_ONBOARD" = 1 ]; then
        say "Next, set it up:  $RUN_CMD setup$RUN_NOTE"
    elif [ "$HAVE_TTY" = 1 ]; then
        say "Starting the setup wizard. You can leave it at any time and run 'edmars setup' later."
        "$LAUNCHER" setup </dev/tty || warn "Setup did not finish. Run 'edmars setup' any time to continue."
    else
        say "No terminal is available for the setup wizard. Run it yourself:  $RUN_CMD setup$RUN_NOTE"
    fi
    say ""
    say "To use EDM-ARS, type:  $RUN_CMD$RUN_NOTE"
}

main "$@"

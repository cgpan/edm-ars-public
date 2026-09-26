# Installing EDM-ARS

The installer sets up EDM-ARS and the `edmars` command inside your own user
account. It needs no administrator rights, changes nothing system-wide and
leaves any Python you already have alone. After it finishes, the setup
wizard (`edmars setup`) walks you through the rest: AI service key, datasets,
PDF tools and the optional reviewer.

**Supported:** Windows 10 (version 1809 or newer) and Windows 11, 64-bit
(tested on Windows 11). **Untested preview:** macOS on Apple Silicon (M1 or
newer), and Linux on x86_64 or arm64 with glibc (Ubuntu, Debian, Fedora and
similar, best effort). The installer is written for them, but it has not yet
been run on a Mac or a Linux computer. **Not supported:** Intel Macs (numba,
llvmlite and scs no longer publish Intel Mac builds of the tested versions),
32-bit Windows, and musl-based Linux such as Alpine.

You need about 6 GB of free disk space (packages, datasets and study
outputs) and an internet connection. 16 GB of memory is recommended.

## Quick install

> **Not yet available.** The download links on this page (quick, careful and
> fully manual install) work only after the first release, v0.1.0, is
> published on [GitHub Releases](https://github.com/cgpan/edm-ars-public/releases).
> Until then they give "404 Not Found". Install from a copy of the
> repository instead (once the `feat/edmars-cli` branch is merged, leave out
> `-b feat/edmars-cli`):
>
> ```powershell
> git clone -b feat/edmars-cli https://github.com/cgpan/edm-ars-public.git
> powershell -ExecutionPolicy Bypass -File .\edm-ars-public\install\install.ps1 -FromLocal .\edm-ars-public
> ```
> ```sh
> git clone -b feat/edmars-cli https://github.com/cgpan/edm-ars-public.git
> sh edm-ars-public/install/install.sh --from-local ./edm-ars-public
> ```

**Windows** (PowerShell):

```powershell
irm https://github.com/cgpan/edm-ars-public/releases/latest/download/install.ps1 | iex
```

**macOS / Linux** (Terminal):

```sh
curl -LsSf https://github.com/cgpan/edm-ars-public/releases/latest/download/install.sh | sh
```

The installer first prints its plan and waits for you to confirm. When it is
done, open a new terminal window and type `edmars`.

## Careful install: download, inspect, then run

Piping a script from the internet into a shell runs it without showing it
to you first. If you (or your IT department) prefer to read it first:

1. Open the [latest release](https://github.com/cgpan/edm-ars-public/releases/latest)
   and download `install.ps1` (Windows) or `install.sh` (macOS/Linux),
   together with `SHA256SUMS`.
2. Check that the script is the published one. The fingerprint printed must
   equal the one on the script's line in `SHA256SUMS`:

   ```powershell
   Get-FileHash .\install.ps1 -Algorithm SHA256        # Windows
   ```
   ```sh
   shasum -a 256 install.sh                            # macOS
   sha256sum install.sh                                # Linux
   ```
3. Read the script. It is plain text, written to be read; every step it
   takes is listed in the plan it prints before doing anything.
4. See what it would do without changing anything:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\install.ps1 -DryRun
   ```
   ```sh
   sh install.sh --dry-run
   ```
5. Run it for real by leaving out `-DryRun` / `--dry-run`.

`-ExecutionPolicy Bypass` applies to that one command only; it does not
change your computer's PowerShell policy.

### Fully manual route

To control every download, also get `edm-ars-<version>.tar.gz` from the
release and point the installer at it. Keep `SHA256SUMS` in the same folder:
the installer checks the archive against it before using it (and says so
when it cannot):

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 -FromLocal .\edm-ars-0.1.0.tar.gz
```
```sh
sh install.sh --from-local ./edm-ars-0.1.0.tar.gz
```

The installer still downloads uv, Python 3.11 and the Python packages (see
below). On Windows the `.tar.gz` is unpacked with the `tar` built into
Windows 10 and 11; you do not need extra software.

## Options

| Windows | macOS / Linux | Effect |
|---|---|---|
| `-Yes` | `--yes` | Do not pause to confirm the plan. |
| `-NoOnboard` | `--no-onboard` | Do not start `edmars setup` at the end. |
| `-Dir D` | `--dir D` | Install into folder `D` instead of the default. |
| `-Version X.Y.Z` | `--version X.Y.Z` | Install that release instead of the latest one. |
| `-FromLocal P` | `--from-local P` | Install from a release `.tar.gz` or a local checkout (also `EDMARS_INSTALL_SOURCE=P`). |
| `-BinDir D` | `--bin-dir D` | Put the `edmars` command in `D`. |
| `-NoModifyPath` | `--no-modify-path` | Do not add the command's folder to your PATH. |
| `-DryRun` | `--dry-run` | Show what would happen; change nothing. |

The Windows script also accepts the `--yes`-style spellings. To pass options
while piping, use
`& ([scriptblock]::Create((irm <url>/install.ps1))) -NoOnboard` on Windows
and `curl -LsSf <url>/install.sh | sh -s -- --no-onboard` elsewhere.

## What goes where

| What | Windows | macOS | Linux |
|---|---|---|---|
| Install folder (`<dir>`) | `%LOCALAPPDATA%\edm-ars` | `~/Library/Application Support/edm-ars` | `~/.local/share/edm-ars` |
| EDM-ARS itself | `<dir>\app\<version>` | `<dir>/app/<version>` | same |
| Its Python packages | `<dir>\venv-<version>` | `<dir>/venv-<version>` | same |
| Private Python 3.11 | `<dir>\python` | `<dir>/python` | same |
| uv (only if you had none) | `<dir>\uv` | `<dir>/uv` | same |
| The `edmars` command | `%USERPROFILE%\.local\bin\edmars.cmd` (and `edmars` beside it, for Git Bash) | `~/.local/bin/edmars` | same |
| Install record | `<dir>\install.json` | `<dir>/install.json` | same |

When you install a new version, the previous one is kept (a study started
before the update may still be running from it); older ones are removed.
The installer never touches your settings, keys, datasets or studies. Keys
are in your system's credential store and studies in the folder you choose.
Datasets and the reviewer (and, on Windows and macOS, your settings) are kept
in your user data folder, which is the same folder as the default install
folder, next to (not inside) `app`, `venv-*`, `python` and `uv` (see `edmars
privacy`). So to remove the program by hand, delete those entries rather than
the whole folder, or run `edmars uninstall` first.

**PATH.** Unless you pass `-NoModifyPath` / `--no-modify-path`, the command's
folder is added to your PATH for your account only: on Windows in your user
environment variables, on macOS and Linux in a clearly marked block at the
end of `~/.profile`, `~/.bashrc` and `~/.zshrc` (and a fish `conf.d` file if
you use fish). Nothing is added when the folder is already there, and
running the installer again does not add it twice.

**Cloud-sync folders.** The installer refuses to install inside OneDrive,
Google Drive, Dropbox or iCloud Drive folders: syncing thousands of package
files is slow and can break the install. If you really want that, name the
folder yourself with `-Dir` / `--dir` and it will warn instead.

## What the installer downloads

| From | What |
|---|---|
| github.com (`cgpan/edm-ars-public` releases) | EDM-ARS itself and `SHA256SUMS` |
| astral.sh / releases.astral.sh | the uv installer (pinned version, fingerprint checked), only if you have no uv 0.8 or newer |
| uv's Python download servers (python-build-standalone builds) | Python 3.11 |
| pypi.org, files.pythonhosted.org | the Python packages pinned in `requirements.lock` |

Nothing is sent except ordinary download requests. EDM-ARS has no telemetry.

## Troubleshooting

- **"running scripts is disabled on this system"** — use the
  `irm ... | iex` line, or run the downloaded file with
  `powershell -ExecutionPolicy Bypass -File .\install.ps1`.
- **"the install folder path is too long for Windows"** — Windows limits
  most programs to 260-character paths, and some package files sit deep in
  the install folder. Choose a short folder, for example `-Dir C:\edm-ars`.
- **macOS: XGBoost and the OpenMP library.** XGBoost's Mac build looks for
  the OpenMP library (`libomp.dylib`) only where Homebrew puts it, and a new
  Mac has no Homebrew. scikit-learn, which EDM-ARS also installs, ships its
  own copy. So when XGBoost finds none, the installer links scikit-learn's
  copy into its private Python (`<dir>/python/cpython-3.11.../lib/libomp.dylib`,
  recorded as `openmp_link` in `install.json`), then checks in one Python
  process that XGBoost and scikit-learn load and share one OpenMP library.
  Nothing outside the install folder changes; no Homebrew or administrator
  rights are needed. `edmars doctor` checks that XGBoost loads.
  - *Why one library:* the link leads to the very file scikit-learn loads,
    so macOS loads it once. Two different OpenMP copies in one process can
    stop a study with "OMP: Error #15".
  - *Risk:* XGBoost then runs on scikit-learn's OpenMP build instead of
    Homebrew's. It provides every OpenMP function XGBoost 2.1.4 uses, and CI
    trains XGBoost and scikit-learn together on it. When you install a new
    EDM-ARS version, the link moves to the new version's packages, so a
    study that was already running from the previous version can stop with
    "OMP: Error #15" at a later step; resume it with `edmars resume`.
  - *If the installer still stops* with "XGBoost needs the OpenMP library",
    install [Homebrew](https://brew.sh), run `brew install libomp`, then run
    the installer again. XGBoost then uses Homebrew's copy while
    scikit-learn keeps its own; `edmars doctor` warns about such a pair.
  - *To undo it,* delete that `libomp.dylib` link; it also goes when you
    delete `<dir>/python`.
- **Behind a proxy** — set `HTTPS_PROXY` before running the installer. uv
  uses it everywhere; on Windows the installer's own downloads follow your
  system proxy settings.
- **"Terminate batch job (Y/N)?" on Windows** after pressing Ctrl+C in
  `edmars` — this is Windows asking about the `edmars.cmd` wrapper. Either
  answer is fine; a running study keeps running.
- **`edmars` is not found after installing** — open a new terminal window.
  Windows and your shell read PATH when a window opens. In Git Bash, if the
  folder has only `edmars.cmd` and no `edmars`, type `edmars.cmd` or run the
  installer again.

Run `edmars doctor` for a full check of your setup.

## Uninstalling

Run `edmars uninstall`: it removes EDM-ARS's settings and stored keys, and
asks separately before touching datasets or studies. It does not remove
TinyTeX, if `edmars setup pdf` installed it, because other programs can use
it: it prints its folder, and you delete that yourself after running
`tlmgr path remove` (not on a Mac, where setup installs TinyTeX without
changing your PATH, so no administrator password is asked for; EDM-ARS
finds it in `~/Library/TinyTeX` itself). R packages from `edmars setup r`
stay in your R library. It cannot delete the
program it is running from, so it ends by listing the program files this
installer created (read from `install.json` in the install folder). Then,
with no `edmars` window open:

1. delete those entries (`app`, `venv-*`, `python`, `uv` if present,
   `install.json` and `versions.txt` in `<dir>`) and the `edmars` command.
   Delete the whole of `<dir>` only if you also want the datasets you kept
   gone: they live in the same folder;
2. undo the PATH change, if the installer made one: on Windows, open
   "Edit environment variables for your account" and remove the
   `...\.local\bin` entry from your user `Path` (only if nothing else of
   yours lives in that folder); on macOS and Linux, delete the block between
   `# >>> edm-ars >>>` and `# <<< edm-ars <<<` in your shell start-up files
   (and the fish file `~/.config/fish/conf.d/edm-ars.fish`, if present).

## For maintainers

- `requirements.lock` pins every package to the versions of an environment
  that passed the full test suite. Regenerate it with
  `python scripts/make_lock.py --tested-python <venv python> --suite-result "<N passed, M skipped>"`
  only after the suite passes in the new environment.
- A release is made by pushing a tag `vX.Y.Z` whose version matches
  `edmars/__init__.py`. The release workflow runs the tests, builds the
  files with `python scripts/make_release.py` (the tarball, both installers
  and `SHA256SUMS`), checks that the installer works from the built tarball,
  and uploads them to the GitHub release.
- The uv version is pinned in both installers together with the SHA-256 of
  its official installer script; update both values together.

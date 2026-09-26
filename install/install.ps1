<#
.SYNOPSIS
    Installs EDM-ARS and the `edmars` command for your Windows user account.

.DESCRIPTION
    The release download addresses below work only after the first release
    (v0.1.0) is published on GitHub Releases. Until then, clone the
    repository and run this script with -FromLocal .\edm-ars-public (see
    install/README.md).

    Quick route (in PowerShell):
        irm https://github.com/cgpan/edm-ars-public/releases/latest/download/install.ps1 | iex

    Careful route (read it first; see install/README.md):
        irm https://github.com/cgpan/edm-ars-public/releases/latest/download/install.ps1 -OutFile install.ps1
        notepad install.ps1
        powershell -ExecutionPolicy Bypass -File install.ps1

    With options when piping:
        & ([scriptblock]::Create((irm https://github.com/cgpan/edm-ars-public/releases/latest/download/install.ps1))) -NoOnboard

    Everything is installed inside your user account; no administrator
    rights are needed and nothing system-wide changes. What goes where:
        <Dir>\app\<version>     EDM-ARS itself (the previous version is kept)
        <Dir>\venv-<version>    its private Python packages
        <Dir>\python            a private Python 3.11 (your own Python is untouched)
        <Dir>\uv                the uv tool and its download cache, only if you
                                do not have uv already
        %USERPROFILE%\.local\bin\edmars.cmd   the command you type
        %USERPROFILE%\.local\bin\edmars       the same command for Git Bash
    Default <Dir>: %LOCALAPPDATA%\edm-ars. To remove it: run `edmars
    uninstall` (settings and keys; it asks about datasets and studies), then
    delete the program files it lists (app, venv-*, python, uv in <Dir>),
    edmars.cmd and edmars. Not all of <Dir>: your data folder is the same
    folder.
    <Dir>\install.json lists everything this installer created.

    This script never closes your PowerShell window: it has no `exit`.
    Works with Windows PowerShell 5.1 and PowerShell 7.

.PARAMETER Yes
    Do not pause to confirm the plan.
.PARAMETER NoOnboard
    Do not start `edmars setup` at the end.
.PARAMETER Dir
    Install into this folder instead of %LOCALAPPDATA%\edm-ars.
.PARAMETER Version
    Install this release (for example 0.1.0) instead of the latest one.
.PARAMETER FromLocal
    Install from a local checkout folder or an edm-ars-X.Y.Z.tar.gz file
    (for testing). Also read from the EDMARS_INSTALL_SOURCE variable.
.PARAMETER BinDir
    Put edmars.cmd (and edmars, for Git Bash) in this folder instead of
    %USERPROFILE%\.local\bin.
.PARAMETER NoModifyPath
    Do not add the command's folder to your user PATH.
.PARAMETER DryRun
    Show what would happen and change nothing.
#>
# The whole installer runs inside one script block: nothing it defines
# (functions, variables) is left behind in a session that ran it with
# `irm | iex`, and a download cut off halfway cannot run half of it.
& {
    # Everything below is scoped to this block, so a user who ran us with
    # `irm | iex` keeps their own session settings afterwards.
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'   # the 5.1 progress bar slows downloads ~10x

    # Read-before-write variables would otherwise resolve to a same-named
    # variable in the caller's session (PowerShell scoping is dynamic), and
    # the cleanup at the end deletes $tmp and $stage.
    $tmp = $null
    $stage = $null
    $pushed = $false
    $savedEnv = @{}

    $Yes = $false
    $NoOnboard = $false
    $NoModifyPath = $false
    $DryRun = $false
    $Help = $false
    $Dir = ''
    $Version = ''
    $FromLocal = ''
    $BinDir = ''

    $EdmarsRepo = 'cgpan/edm-ars-public'
    # uv installs Python and the packages. Pinned: the installer script at
    # this URL is fixed for a given uv version, and its SHA-256 is checked.
    $UvVersion = '0.10.6'
    $UvInstallerSha256 = 'e6046a0282ab848c48566097f0bac64074d32cda1d504279e079470cb99cbfd5'
    $UvMinVersion = [version]'0.8.0'
    $PythonSeries = '3.11'
    $MinFreeGB = 6
    $RamAdvisedGB = 16
    $LauncherMark = 'EDM-ARS command, written by the EDM-ARS installer'

    # ---- small helpers ------------------------------------------------------
    function Say([string]$Text) { Write-Host $Text }
    function Step([string]$Num, [string]$Text) { Write-Host ''; Write-Host "[$Num] $Text" -ForegroundColor Cyan }
    function Warn([string]$Text) { Write-Host "  ! $Text" -ForegroundColor Yellow }
    function Stop-Install([string]$Text) {
        Write-Host ''
        Write-Host "The installation stopped: $Text" -ForegroundColor Red
        # throw, never exit: exit would close the window of a user who
        # piped this script into iex.
        throw 'EDM-ARS installation did not finish (see the message above).'
    }

    # Run a program, show its output, and stop with a clear message if it
    # fails. Windows PowerShell 5.1 turns a native program's stderr lines
    # into errors, which with ErrorActionPreference=Stop would abort on the
    # first progress line uv prints; so errors are relaxed here and the
    # exit code decides.
    function Invoke-Checked([string]$What, [string]$Exe, [string[]]$Arguments) {
        # A missing program is only a non-terminating error once errors are
        # relaxed, and $LASTEXITCODE would still hold the previous program's
        # code; so check first.
        if (-not (Get-Command $Exe -ErrorAction SilentlyContinue)) {
            Stop-Install "$What failed: $Exe was not found."
        }
        $code = $null
        $saved = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            & $Exe @Arguments 2>&1 | ForEach-Object {
                if ($_ -is [System.Management.Automation.ErrorRecord]) {
                    Write-Host $_.Exception.Message
                } else {
                    Write-Host $_
                }
            }
            $code = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $saved
        }
        if ($code -ne 0) { Stop-Install "$What failed (exit code $code; see the messages above)." }
    }

    # Capture one line of a program's output, or $null on failure.
    function Get-NativeOutput([string]$Exe, [string[]]$Arguments) {
        $saved = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $out = & $Exe @Arguments 2>$null
            if ($LASTEXITCODE -ne 0) { return $null }
            return (@($out) | Where-Object { $_ } | Select-Object -Last 1)
        } catch {
            return $null
        } finally {
            $ErrorActionPreference = $saved
        }
    }

    # Environment variables we set for child programs are restored at the
    # end (from $savedEnv), so an `irm | iex` session is not left with them.
    function Set-TempEnv([string]$Name, [string]$Value) {
        if (-not $savedEnv.ContainsKey($Name)) {
            $savedEnv[$Name] = [Environment]::GetEnvironmentVariable($Name, 'Process')
        }
        [Environment]::SetEnvironmentVariable($Name, $Value, 'Process')
    }

    function Get-Sha256([string]$Path) {
        return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    }

    # Download URL to DEST. A file:// URL or a plain path is copied, which
    # lets EDMARS_RELEASE_BASE_URL point at a local dist\ folder for testing.
    # Only https:// is downloaded, as in install.sh: over plain http:// the
    # archive and its SHA256SUMS could both be swapped on the way.
    function Get-Download([string]$Url, [string]$Dest) {
        if ($Url -match '^file:') {
            Copy-Item -LiteralPath ([uri]$Url).LocalPath -Destination $Dest -Force
        } elseif ($Url -match '^https://') {
            Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
        } elseif ($Url -match '^[A-Za-z][A-Za-z0-9+.-]*://') {
            throw "only https:// addresses are downloaded, not $Url"
        } else {
            Copy-Item -LiteralPath $Url -Destination $Dest -Force
        }
    }

    function Join-Url([string]$Base, [string]$Name) {
        if ($Base -match '^(https?|file):') { return ($Base.TrimEnd('/') + '/' + $Name) }
        return (Join-Path $Base $Name)
    }

    function Resolve-FullPath([string]$Path) {
        if ($Path.StartsWith('~')) { $Path = $HOME + $Path.Substring(1) }
        return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path).TrimEnd('\')
    }

    function Test-UnderPath([string]$Path, [string]$Root) {
        if (-not $Root) { return $false }
        $p = $Path.TrimEnd('\') + '\'
        $r = $Root.TrimEnd('\') + '\'
        return $p.StartsWith($r, [StringComparison]::OrdinalIgnoreCase)
    }

    # The raw user PATH with DIR appended, or $null when DIR is already on
    # it. Entries are compared after expanding %VARIABLES%, case-insensitively
    # and ignoring a trailing backslash; the user's own entries are kept
    # exactly as stored. In an expandable (REG_EXPAND_SZ) value a folder under
    # the profile is added as %USERPROFILE%\..., like Windows' own entries.
    function Merge-UserPath([string]$Raw, [string]$Dir, [bool]$Expandable) {
        foreach ($entry in ($Raw -split ';')) {
            if (-not $entry) { continue }
            if ([Environment]::ExpandEnvironmentVariables($entry).TrimEnd('\') -ieq $Dir.TrimEnd('\')) { return $null }
        }
        $add = $Dir
        $profileDir = $env:USERPROFILE
        if ($Expandable -and $profileDir -and (Test-UnderPath $Dir $profileDir)) {
            $add = '%USERPROFILE%' + $Dir.Substring($profileDir.TrimEnd('\').Length)
        }
        if ($Raw.Trim(';')) { return ($Raw.TrimEnd(';') + ';' + $add) }
        return $add
    }

    # Name of the cloud-sync service whose folder contains PATH, if any.
    function Get-SyncProvider([string]$Path) {
        foreach ($name in @('OneDrive', 'OneDriveCommercial', 'OneDriveConsumer')) {
            $root = [Environment]::GetEnvironmentVariable($name)
            if ($root -and (Test-UnderPath $Path $root)) { return 'OneDrive' }
        }
        foreach ($part in ($Path -split '[\\/]')) {
            if ($part -like 'OneDrive*') { return 'OneDrive' }
            if ($part -like 'Dropbox*') { return 'Dropbox' }
            if ($part -eq 'My Drive' -or $part -eq 'Google Drive' -or $part -like 'GoogleDrive*') { return 'Google Drive' }
            if ($part -eq 'iCloudDrive' -or $part -eq 'iCloud Drive') { return 'iCloud Drive' }
        }
        try {
            $qualifier = [System.IO.Path]::GetPathRoot($Path)
            if ($qualifier -and (Test-Path -LiteralPath (Join-Path $qualifier '.shortcut-targets-by-id'))) {
                return 'Google Drive'
            }
        } catch { }
        return $null
    }

    function Get-ExistingParent([string]$Path) {
        $p = $Path
        while ($p -and -not (Test-Path -LiteralPath $p -PathType Container)) {
            $parent = Split-Path -Parent $p
            if (-not $parent -or $parent -eq $p) { break }
            $p = $parent
        }
        return $p
    }

    # Paths written into edmars.cmd. Under %LOCALAPPDATA% or %USERPROFILE%
    # they are written with the variable, so a user name with non-ASCII
    # letters (which cmd.exe reads in the console code page) still works.
    function ConvertTo-CmdPath([string]$Path) {
        foreach ($var in @('LOCALAPPDATA', 'USERPROFILE')) {
            $root = [Environment]::GetEnvironmentVariable($var)
            if ($root -and (Test-UnderPath $Path $root)) {
                $rest = $Path.Substring($root.TrimEnd('\').Length)
                return ('%' + $var + '%' + $rest.Replace('%', '%%'))
            }
        }
        return $Path.Replace('%', '%%')
    }

    # The same command for Git Bash and other POSIX shells on Windows. bash
    # does not use PATHEXT, so typing `edmars` there never finds edmars.cmd.
    # cmd.exe and PowerShell keep running edmars.cmd (they look for the
    # .cmd first). Paths are single-quoted for sh; the program path uses
    # forward slashes, which Git Bash runs as they are.
    function Get-ShLauncherText([string]$Mark, [string]$Ver, [string]$AppDir, [string]$Uv, [string]$VenvPy) {
        $q = { param([string]$t) "'" + $t.Replace("'", "'\''") + "'" }
        $lines = @(
            '#!/bin/sh',
            "# $Mark (EDM-ARS $Ver).",
            '# For Git Bash, which does not run edmars.cmd when you type edmars. Re-run the installer to update it.',
            ('EDMARS_APP_ROOT=' + (& $q $AppDir)),
            'export EDMARS_APP_ROOT',
            'PYTHONUTF8=1',
            'export PYTHONUTF8',
            'unset PYTHONHOME PYTHONPATH',
            ('if [ -e ' + (& $q ($Uv -replace '\\', '/')) + ' ]; then'),
            ('    EDMARS_UV=' + (& $q $Uv)),
            '    export EDMARS_UV',
            'fi',
            ('exec ' + (& $q ($VenvPy -replace '\\', '/')) + ' -P -m edmars "$@"')
        )
        return ($lines -join "`n") + "`n"
    }

    # Copy a checkout without git history, caches, raw data, run outputs or
    # local secrets.
    $ExcludeAnywhere = @('.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'node_modules', '.env')
    $ExcludeTop = @('output', 'data', 'dist', 'cache', '.venv', 'venv', 'ideas', 'files')
    function Test-Excluded([string]$Rel, [string]$Name) {
        if ($ExcludeAnywhere -contains $Name) { return $true }
        if ($Name -like '*.pyc' -or $Name -like '*.env') { return $true }
        $parts = $Rel -split '/'
        if ($parts.Count -eq 1 -and (($ExcludeTop -contains $Name) -or ($Name -like 'tmp_orch_test*'))) { return $true }
        if ($parts.Count -eq 3 -and $parts[0] -eq 'runs' -and $parts[2] -like 'output*') { return $true }
        return $false
    }
    function Copy-Tree([string]$Src, [string]$Dst, [string]$Rel) {
        $null = New-Item -ItemType Directory -Force -Path $Dst
        foreach ($item in (Get-ChildItem -LiteralPath $Src -Force)) {
            $name = $item.Name
            if ($Rel) { $childRel = "$Rel/$name" } else { $childRel = $name }
            if (Test-Excluded $childRel $name) { continue }
            if ($item.PSIsContainer) {
                if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { continue }
                Copy-Tree $item.FullName (Join-Path $Dst $name) $childRel
            } else {
                Copy-Item -LiteralPath $item.FullName -Destination (Join-Path $Dst $name) -Force
            }
        }
    }

    # ---- arguments ----------------------------------------------------------------
    # PowerShell spelling (-Yes, -Dir X) and GNU spelling (--yes, --dir X,
    # --dir=X) are both accepted; names are case-insensitive.
    $argv = @($args)
    $i = 0
    while ($i -lt $argv.Count) {
        $arg = [string]$argv[$i]
        $value = $null
        if ($arg -match '^(--?[A-Za-z-]+)[=:](.*)$') { $arg = $Matches[1]; $value = $Matches[2] }
        $takesValue = $false
        switch ($arg) {
            { $_ -in @('-Yes', '--yes', '-y') } { $Yes = $true }
            { $_ -in @('-NoOnboard', '--no-onboard') } { $NoOnboard = $true }
            { $_ -in @('-NoModifyPath', '--no-modify-path') } { $NoModifyPath = $true }
            { $_ -in @('-DryRun', '--dry-run') } { $DryRun = $true }
            { $_ -in @('-Help', '--help', '-h', '-?') } { $Help = $true }
            { $_ -in @('-Dir', '--dir', '-Version', '--version', '-FromLocal', '--from-local', '-BinDir', '--bin-dir') } { $takesValue = $true }
            default { Stop-Install "unknown option '$arg' (see -Help)." }
        }
        if ($takesValue) {
            if ($null -eq $value) {
                $i++
                if ($i -ge $argv.Count) { Stop-Install "$arg needs a value." }
                $value = [string]$argv[$i]
            }
            switch ($arg.TrimStart('-').Replace('-', '')) {
                'dir' { $Dir = $value }
                'version' { $Version = $value }
                'fromlocal' { $FromLocal = $value }
                'bindir' { $BinDir = $value }
            }
        }
        $i++
    }
    if ($Help) {
        Say 'EDM-ARS installer for Windows.'
        Say ''
        Say 'Usage: powershell -ExecutionPolicy Bypass -File install.ps1 [options]'
        Say ''
        Say 'Options (PowerShell or GNU spelling):'
        Say '  -Yes            --yes             do not pause to confirm the plan'
        Say '  -NoOnboard      --no-onboard      do not start "edmars setup" at the end'
        Say '  -Dir DIR        --dir DIR         install into DIR (default %LOCALAPPDATA%\edm-ars)'
        Say '  -Version X.Y.Z  --version X.Y.Z   install this release instead of the latest one'
        Say '  -FromLocal P    --from-local P    install from a local checkout or edm-ars-X.Y.Z.tar.gz'
        Say '  -BinDir DIR     --bin-dir DIR     put edmars.cmd in DIR (default %USERPROFILE%\.local\bin)'
        Say '  -NoModifyPath   --no-modify-path  do not add the command folder to your user PATH'
        Say '  -DryRun         --dry-run         show what would happen and change nothing'
        return
    }

    $dirExplicit = $false
    if (-not $Dir -and $env:EDMARS_INSTALL_DIR) { $Dir = $env:EDMARS_INSTALL_DIR }
    if ($Dir) { $dirExplicit = $true }
    if (-not $FromLocal -and $env:EDMARS_INSTALL_SOURCE) { $FromLocal = $env:EDMARS_INSTALL_SOURCE }
    if (-not $BinDir -and $env:EDMARS_BIN_DIR) { $BinDir = $env:EDMARS_BIN_DIR }

    try {
        # ---- computer check -------------------------------------------------------
        $interactive = [Environment]::UserInteractive -and -not [Console]::IsInputRedirected
        $isWindows_ = [Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT
        if (-not $isWindows_) { Stop-Install 'this script is for Windows. On macOS or Linux use install.sh.' }
        $osVersion = [Environment]::OSVersion.Version
        $supported = $true
        $platformNote = ''
        if ($osVersion -lt [version]'10.0.17763') {
            $supported = $false
            $platformNote = 'Windows 10 version 1809 or newer is needed.'
        }
        $arch = $env:PROCESSOR_ARCHITEW6432
        if (-not $arch) { $arch = $env:PROCESSOR_ARCHITECTURE }
        $pyRequest = $PythonSeries
        switch ($arch) {
            'AMD64' { $platform = 'Windows (64-bit)' }
            'ARM64' {
                $platform = 'Windows on Arm'
                # Most scientific packages publish no Windows-on-Arm builds;
                # the x64 Python runs under Windows' built-in emulation.
                $pyRequest = "cpython-$PythonSeries-windows-x86_64-none"
                $platformNote = 'On Windows on Arm, EDM-ARS uses the 64-bit Intel/AMD Python under emulation; it works but runs slower.'
            }
            default {
                $platform = "Windows ($arch)"
                $supported = $false
                $platformNote = '32-bit Windows is not supported.'
            }
        }
        $tarExe = Join-Path $env:SystemRoot 'System32\tar.exe'
        if (-not (Test-Path -LiteralPath $tarExe)) {
            $supported = $false
            $platformNote = "Windows' built-in tar.exe is missing (it ships with Windows 10 version 1803 and newer)."
        }
        if (-not $supported -and -not $DryRun) { Stop-Install $platformNote }

        if (-not $env:LOCALAPPDATA -or -not $env:USERPROFILE) { Stop-Install 'LOCALAPPDATA or USERPROFILE is not set.' }
        if ($Dir) { $base = Resolve-FullPath $Dir } else { $base = Join-Path $env:LOCALAPPDATA 'edm-ars' }
        if ($BinDir) { $bin = Resolve-FullPath $BinDir } else { $bin = Join-Path $env:USERPROFILE '.local\bin' }

        $sync = Get-SyncProvider $base
        if ($sync) {
            if ($dirExplicit -or $DryRun) {
                Warn "$base is inside a $sync folder. Syncing thousands of package files is slow and can corrupt the install; a local folder is strongly recommended."
            } else {
                Stop-Install "the install folder $base is inside a $sync folder. Choose a local folder with -Dir."
            }
        }

        $freeGB = $null
        $freeParent = Get-ExistingParent $base
        try {
            $drive = New-Object System.IO.DriveInfo ([System.IO.Path]::GetPathRoot($freeParent))
            $freeGB = [math]::Floor($drive.AvailableFreeSpace / 1GB)
        } catch { $freeGB = $null }

        $ramGB = $null
        try {
            $ramGB = [math]::Round((Get-CimInstance -ClassName Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
        } catch {
            try { $ramGB = [math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB) } catch { $ramGB = $null }
        }

        # ---- where the code comes from ---------------------------------------------
        $sourceKind = 'github-release'
        $ver = ''
        $releaseBase = ''
        $requested = $Version
        if ($requested.StartsWith('v')) { $requested = $requested.Substring(1) }
        if ($requested -eq 'latest') { $requested = '' }
        if ($FromLocal) {
            $FromLocal = Resolve-FullPath $FromLocal
            if (Test-Path -LiteralPath $FromLocal -PathType Container) {
                $sourceKind = 'local-folder'
                if (-not (Test-Path -LiteralPath (Join-Path $FromLocal 'src\main.py'))) {
                    Stop-Install "$FromLocal does not look like an EDM-ARS checkout (no src\main.py)."
                }
                $ver = $requested
                $initPy = Join-Path $FromLocal 'edmars\__init__.py'
                if (-not $ver -and (Test-Path -LiteralPath $initPy)) {
                    $m = Select-String -LiteralPath $initPy -Pattern '^__version__\s*=\s*[''"]([^''"]+)[''"]' | Select-Object -First 1
                    if ($m) { $ver = $m.Matches[0].Groups[1].Value }
                }
                if (-not $ver) { $ver = 'local' }
            } elseif (Test-Path -LiteralPath $FromLocal -PathType Leaf) {
                $sourceKind = 'local-archive'
                $leaf = Split-Path -Leaf $FromLocal
                if ($leaf -notmatch '^edm-ars-(.+)\.tar\.gz$') { Stop-Install "$FromLocal is not an edm-ars-<version>.tar.gz release archive." }
                $ver = $Matches[1]
            } else {
                Stop-Install "-FromLocal $FromLocal does not exist."
            }
            $sourceText = $FromLocal
        } else {
            $ver = $requested
            if ($env:EDMARS_RELEASE_BASE_URL) {
                $releaseBase = $env:EDMARS_RELEASE_BASE_URL.TrimEnd('/')
                if (($releaseBase -match '^[A-Za-z][A-Za-z0-9+.-]*://') -and ($releaseBase -notmatch '^(https://|file:)')) {
                    Stop-Install "EDMARS_RELEASE_BASE_URL must start with https:// (or be a local folder or a file:// address). A download over plain http:// could be altered on the way, so nothing was downloaded."
                }
            } elseif ($requested) {
                $releaseBase = "https://github.com/$EdmarsRepo/releases/download/v$requested"
            } else {
                $releaseBase = "https://github.com/$EdmarsRepo/releases/latest/download"
            }
            $sourceText = $releaseBase
        }
        if ($ver -and ($ver -notmatch '^[0-9A-Za-z][0-9A-Za-z.+-]*$')) { Stop-Install "'$ver' is not a valid version." }

        # Windows' 260-character path limit. uv writes long paths fine, but
        # unless long paths are enabled Python cannot load a file whose full
        # path is longer, and the longest files it loads from the package
        # folder are ~113 characters below it (a numpy DLL, a scikit-learn
        # module). Found the hard way: a deep test folder installed cleanly
        # and then failed with "No module named sklearn...". 140 leaves room.
        $longPaths = $false
        try {
            $fs = Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -ErrorAction Stop
            $longPaths = ($fs.LongPathsEnabled -eq 1)
        } catch { $longPaths = $false }
        $verLen = 10
        if ($ver) { $verLen = $ver.Length }
        $venvLen = $base.Length + '\venv-'.Length + $verLen
        if (-not $longPaths -and $venvLen -gt 140) {
            $msg = "the install folder path is too long for Windows ($($base.Length) characters): some package files would pass Windows' 260-character limit and fail to load. Choose a shorter folder with -Dir, for example C:\edm-ars."
            if ($DryRun) { Warn $msg } else { Stop-Install $msg }
        }

        # ---- uv -----------------------------------------------------------------
        $uv = $null
        $uvFound = $null
        $uvPrivate = $false
        if (-not $env:EDMARS_FORCE_PRIVATE_UV) {
            $candidates = @()
            $cmd = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($cmd) { $candidates += $cmd.Source }
            $candidates += (Join-Path $env:USERPROFILE '.local\bin\uv.exe')
            $candidates += (Join-Path $env:USERPROFILE '.cargo\bin\uv.exe')
            $candidates += (Join-Path $base 'uv\uv.exe')
            foreach ($c in $candidates) {
                if (-not (Test-Path -LiteralPath $c -PathType Leaf)) { continue }
                $line = Get-NativeOutput $c @('--version')
                if (-not $line) { continue }
                $text = ([string]$line).Split(' ')
                if ($text.Count -lt 2) { continue }
                $parsed = $null
                if (-not [version]::TryParse(($text[1] -replace '[^0-9.].*$', ''), [ref]$parsed)) { continue }
                if ($parsed -ge $UvMinVersion) { $uv = $c; $uvFound = $text[1]; break }
            }
        }

        # ---- the plan -------------------------------------------------------------
        $facts = $platform
        if ($ramGB) { $facts += ", $ramGB GB memory" }
        if ($null -ne $freeGB) { $facts += ", $freeGB GB free" }
        Say ''
        Say 'EDM-ARS installer'
        Say '================='
        Say "Computer:      $facts"
        if ($platformNote) { Warn $platformNote }
        Say "Install into:  $base"
        Say "Command in:    $(Join-Path $bin 'edmars.cmd') (and edmars, for Git Bash)"
        if ($FromLocal) {
            Say "EDM-ARS from:  $sourceText (local copy, version label '$ver')"
        } else {
            $shown = $ver
            if (-not $shown) { $shown = 'latest release' }
            Say "EDM-ARS from:  $sourceText ($shown)"
        }
        Say ''
        Say 'This will:'
        Say '  1. Check this computer (system, free disk space, memory).'
        if ($uv) {
            Say "  2. Use the uv tool you already have ($uv, version $uvFound)."
        } else {
            Say "  2. Download the uv tool $UvVersion from astral.sh into $(Join-Path $base 'uv')"
            Say '     (checked against a fixed SHA-256 fingerprint; not added to your PATH).'
        }
        Say "  3. Install a private Python $PythonSeries (any Python you already have is left alone)."
        if ($FromLocal) {
            Say "  4. Copy EDM-ARS from $FromLocal."
        } else {
            Say '  4. Download EDM-ARS from GitHub and check its SHA-256 fingerprint.'
        }
        Say '  5. Install EDM-ARS and the packages it needs (about 1.5 GB).'
        Say "  6. Create the command $(Join-Path $bin 'edmars.cmd') (and $(Join-Path $bin 'edmars') for Git Bash)."
        if ($NoModifyPath) {
            Say '  7. Leave your PATH alone (-NoModifyPath).'
        } else {
            Say "  7. Add $bin to your user PATH (your account only)."
        }
        if ($NoOnboard) {
            Say "  8. Stop there (-NoOnboard); run 'edmars setup' when you are ready."
        } else {
            Say "  8. Start the setup wizard ('edmars setup')."
        }
        Say ''
        Say 'No administrator rights are needed. To remove it later, run ''edmars uninstall'''
        Say "(your settings and keys), then delete the program files it lists from $base and the command."

        if (($null -ne $freeGB) -and ($freeGB -lt $MinFreeGB)) {
            $msg = "only $freeGB GB free on the drive holding $freeParent; EDM-ARS needs at least $MinFreeGB GB (packages, data and outputs)."
            if ($DryRun) { Warn $msg } else { Stop-Install "$msg Free some space or choose another drive with -Dir." }
        }
        if ($ramGB -and ($ramGB -lt $RamAdvisedGB)) {
            Warn "This computer has $ramGB GB of memory. $RamAdvisedGB GB is recommended; with less, large datasets such as HSLS:09 load slowly and can run out of memory."
        }

        if ($DryRun) {
            Say ''
            Say 'Dry run: nothing was downloaded or changed.'
            return
        }

        if (-not $Yes) {
            if (-not $interactive) {
                Stop-Install 'there is no console to ask for confirmation. Re-run with -Yes to install without asking.'
            }
            $answer = Read-Host "`nContinue? [Y/n]"
            if ($answer -and ($answer -notmatch '^(y|yes)$')) {
                Say 'Nothing was installed.'
                return
            }
        }

        # ---- set up ------------------------------------------------------------------
        $appParent = Join-Path $base 'app'
        $null = New-Item -ItemType Directory -Force -Path $appParent
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ('edmars-install-' + [guid]::NewGuid().ToString('N').Substring(0, 12))
        $null = New-Item -ItemType Directory -Force -Path $tmp
        $stage = Join-Path $appParent ('.staging-' + $PID)
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        # uv reads configuration from the current folder; run it from a neutral one.
        Push-Location -LiteralPath $tmp
        $pushed = $true
        # Python variables from the calling session must not leak into the
        # private interpreter we are about to build and test.
        foreach ($name in @('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'CONDA_PREFIX')) { Set-TempEnv $name $null }

        # ---- 2. uv ------------------------------------------------------------------
        Step '2/8' 'Getting uv'
        if ($uv) {
            Say "Using $uv ($uvFound)."
            # A uv this installer put in the install folder earlier is still ours.
            $uvPrivate = Test-UnderPath $uv $base
        } else {
            $uvInstaller = Join-Path $tmp 'uv-installer.ps1'
            try {
                Get-Download "https://astral.sh/uv/$UvVersion/install.ps1" $uvInstaller
            } catch {
                Stop-Install "could not download the uv installer ($($_.Exception.Message)). Check your internet connection."
            }
            if ((Get-Sha256 $uvInstaller) -ne $UvInstallerSha256) {
                Stop-Install 'the uv installer did not match its expected fingerprint, so it was not run. You can install uv yourself (https://docs.astral.sh/uv/) and run this installer again.'
            }
            # UV_UNMANAGED_INSTALL: install into that folder only, without
            # touching PATH or uv's self-update receipt. Run it as a separate
            # PowerShell so nothing it does can affect this session.
            Set-TempEnv 'UV_UNMANAGED_INSTALL' (Join-Path $base 'uv')
            $psExe = Join-Path $PSHOME 'powershell.exe'
            if (-not (Test-Path -LiteralPath $psExe)) { $psExe = Join-Path $PSHOME 'pwsh.exe' }
            Invoke-Checked 'The uv installer' $psExe @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $uvInstaller)
            Set-TempEnv 'UV_UNMANAGED_INSTALL' $null
            $uv = Join-Path $base 'uv\uv.exe'
            $uvPrivate = $true
            if (-not (Test-Path -LiteralPath $uv)) { Stop-Install "uv was not found at $uv after installing it." }
        }
        if ($uvPrivate) {
            # uv's download cache would otherwise stay in %LOCALAPPDATA%\uv\cache
            # after the program is deleted. In the uv folder it goes with it.
            # A uv of your own keeps using its own cache.
            Set-TempEnv 'UV_CACHE_DIR' (Join-Path $base 'uv\cache')
        }

        # ---- 3. Python --------------------------------------------------------------
        Step '3/8' "Installing a private Python $PythonSeries"
        Set-TempEnv 'UV_PYTHON_INSTALL_DIR' (Join-Path $base 'python')
        Invoke-Checked "Installing Python $PythonSeries" $uv @('python', 'install', $pyRequest, '--no-bin', '--no-registry')
        $basePy = Get-NativeOutput $uv @('python', 'find', '--managed-python', '--no-project', $pyRequest)
        if (-not $basePy) { Stop-Install "the private Python $PythonSeries was installed but cannot be found." }

        # ---- 4. EDM-ARS -------------------------------------------------------------
        if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
        $null = New-Item -ItemType Directory -Force -Path $stage
        if ($sourceKind -eq 'local-folder') {
            Step '4/8' "Copying EDM-ARS from $FromLocal"
            $newSrc = Join-Path $stage 'src'
            Copy-Tree $FromLocal $newSrc ''
        } else {
            $sums = Join-Path $tmp 'SHA256SUMS'
            if ($sourceKind -eq 'local-archive') {
                Step '4/8' "Checking $FromLocal"
                $tarball = $FromLocal
                $tarName = Split-Path -Leaf $FromLocal
                $sumsBeside = Join-Path (Split-Path -Parent $FromLocal) 'SHA256SUMS'
                if (Test-Path -LiteralPath $sumsBeside) {
                    Copy-Item -LiteralPath $sumsBeside -Destination $sums
                } else {
                    Warn "No SHA256SUMS next to $FromLocal, so its fingerprint cannot be checked."
                    Set-Content -LiteralPath $sums -Value '' -Encoding Ascii
                }
            } else {
                Step '4/8' 'Downloading EDM-ARS'
                try {
                    Get-Download (Join-Url $releaseBase 'SHA256SUMS') $sums
                } catch {
                    $hint = ''
                    if ($requested) { $hint = " and that version $requested exists" }
                    Stop-Install "could not download SHA256SUMS from $releaseBase ($($_.Exception.Message)). Check your internet connection$hint."
                }
                $tarName = $null
                foreach ($line in (Get-Content -LiteralPath $sums)) {
                    if ($line -match '^[0-9a-fA-F]{64}[ *]{1,2}(edm-ars-[0-9A-Za-z.+-]+\.tar\.gz)\s*$') { $tarName = $Matches[1]; break }
                }
                if (-not $tarName) { Stop-Install 'the release has no edm-ars-<version>.tar.gz listed in SHA256SUMS.' }
                $found = $tarName.Substring(8, $tarName.Length - 8 - 7)
                if ($requested -and ($found -ne $requested)) { Stop-Install "asked for version $requested but the release lists $tarName." }
                $ver = $found
                if ($ver -notmatch '^[0-9A-Za-z][0-9A-Za-z.+-]*$') { Stop-Install "'$ver' is not a valid version." }
                if (-not $env:EDMARS_RELEASE_BASE_URL -and -not $requested) {
                    # Fetch the archive from the release the checksums came from.
                    $tarUrl = "https://github.com/$EdmarsRepo/releases/download/v$ver/$tarName"
                } else {
                    $tarUrl = Join-Url $releaseBase $tarName
                }
                Say "Release found: EDM-ARS $ver"
                $tarball = Join-Path $tmp $tarName
                try {
                    Get-Download $tarUrl $tarball
                } catch {
                    Stop-Install "could not download $tarUrl ($($_.Exception.Message))."
                }
            }
            $expected = $null
            foreach ($line in (Get-Content -LiteralPath $sums)) {
                if ($line -match '^([0-9a-fA-F]{64})[ *]{1,2}(\S+)\s*$' -and $Matches[2] -eq $tarName) { $expected = $Matches[1].ToLowerInvariant(); break }
            }
            if ((-not $expected) -and ($sourceKind -eq 'local-archive') -and (Get-Content -LiteralPath $sums | Where-Object { $_.Trim() })) {
                Warn "The SHA256SUMS next to $FromLocal does not list $tarName, so its fingerprint cannot be checked."
            }
            if ($expected) {
                $actual = Get-Sha256 $tarball
                if ($actual -ne $expected) {
                    Stop-Install "$tarName does not match its SHA-256 fingerprint (expected $expected, got $actual). The download may be damaged or altered, so EDM-ARS was not installed."
                }
                Say "Fingerprint OK ($expected)."
            } elseif ($sourceKind -eq 'github-release') {
                Stop-Install "SHA256SUMS does not list $tarName."
            }
            Invoke-Checked "Unpacking $tarName" $tarExe @('-xzf', $tarball, '-C', $stage)
            $newSrc = Join-Path $stage "edm-ars-$ver"
            if (-not (Test-Path -LiteralPath $newSrc -PathType Container)) { Stop-Install "$tarName does not contain the folder edm-ars-$ver." }
        }
        if (-not (Test-Path -LiteralPath (Join-Path $newSrc 'src\main.py'))) { Stop-Install 'the downloaded copy has no src\main.py.' }
        if (-not (Test-Path -LiteralPath (Join-Path $newSrc 'requirements.lock'))) {
            Stop-Install 'this copy of EDM-ARS has no requirements.lock, so its packages cannot be installed reproducibly.'
        }
        if (-not (Test-Path -LiteralPath (Join-Path $newSrc 'edmars\__main__.py'))) {
            Warn "This copy has no edmars package (edmars\__main__.py); the 'edmars' command will not start."
        }

        $appDir = Join-Path $appParent $ver
        $venv = Join-Path $base "venv-$ver"
        $venvPy = Join-Path $venv 'Scripts\python.exe'
        $versionsFile = Join-Path $base 'versions.txt'
        $previous = $null
        if (Test-Path -LiteralPath $versionsFile) {
            $previous = [string](Get-Content -LiteralPath $versionsFile | Where-Object { $_ } | Select-Object -Last 1)
        }
        if (Test-Path -LiteralPath $appDir) {
            try { Remove-Item -LiteralPath $appDir -Recurse -Force } catch {
                Stop-Install "cannot replace $appDir ($($_.Exception.Message)). Close any running 'edmars' window and try again."
            }
        }
        Move-Item -LiteralPath $newSrc -Destination $appDir
        Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue

        # ---- 5. packages ------------------------------------------------------------
        Step '5/8' 'Installing packages (this can take several minutes)'
        if (Test-Path -LiteralPath $venv) {
            try { Remove-Item -LiteralPath $venv -Recurse -Force } catch {
                Stop-Install "cannot replace $venv ($($_.Exception.Message)). Close any running 'edmars' window and try again."
            }
        }
        Invoke-Checked 'Creating the Python environment' $uv @('venv', '--quiet', '--no-project', '--python', $basePy, $venv)
        $pipArgs = @('pip', 'install', '--python', $venvPy, '-r', (Join-Path $appDir 'requirements.lock'))
        $cliReq = Join-Path $appDir 'requirements-cli.txt'
        if (Test-Path -LiteralPath $cliReq) { $pipArgs += @('-r', $cliReq) }
        Invoke-Checked 'Installing the packages' $uv $pipArgs
        Say 'Checking that the main packages load...'
        Invoke-Checked 'Loading the main packages' $venvPy @('-c', 'import numpy, pandas, scipy, sklearn, matplotlib, xgboost, shap, fitz, yaml, requests')

        # EDM-ARS itself is not a pip package: the app folder goes on the
        # environment's import path through a .pth file, so `python -m
        # edmars` (and the CLI's `import src...`) find this version's code
        # wherever the user's current folder is.
        $sitePkgs = Get-NativeOutput $venvPy @('-c', "import sysconfig; print(sysconfig.get_paths()['purelib'])")
        if (-not $sitePkgs -or -not (Test-Path -LiteralPath $sitePkgs -PathType Container)) {
            Stop-Install 'could not find the package folder of the new Python environment.'
        }
        $pth = Join-Path $sitePkgs 'edm_ars_app.pth'
        if ($appDir -match '^[\x20-\x7E]+$') {
            $pthText = $appDir + "`n"
        } else {
            # Python 3.11 reads .pth files in the locale encoding, so a
            # non-ASCII path is written as an ASCII-only escaped literal.
            $sb = New-Object System.Text.StringBuilder
            foreach ($ch in $appDir.ToCharArray()) {
                $code = [int]$ch
                if ($ch -eq '\') { $null = $sb.Append('\\') }
                elseif ($ch -eq "'") { $null = $sb.Append("\'") }
                elseif ($code -lt 32 -or $code -gt 126) { $null = $sb.Append(('\u{0:x4}' -f $code)) }
                else { $null = $sb.Append($ch) }
            }
            $lit = "'" + $sb.ToString() + "'"
            $pthText = "import sys; $lit in sys.path or sys.path.append($lit)`n"
        }
        [System.IO.File]::WriteAllText($pth, $pthText, (New-Object System.Text.UTF8Encoding $false))

        # ---- 6. the command ------------------------------------------------------------
        Step '6/8' 'Creating the edmars command'
        $null = New-Item -ItemType Directory -Force -Path $bin
        $launcher = Join-Path $bin 'edmars.cmd'
        if ((Test-Path -LiteralPath $launcher) -and -not (Select-String -LiteralPath $launcher -SimpleMatch $LauncherMark -Quiet)) {
            Stop-Install "$launcher already exists and was not made by this installer; move it away or choose another folder with -BinDir."
        }
        $cmdLines = @(
            '@echo off',
            "rem $LauncherMark (EDM-ARS $ver).",
            'rem Re-run the installer to update it. install.json in the install folder lists everything it created.',
            'setlocal',
            ('set "EDMARS_APP_ROOT=' + (ConvertTo-CmdPath $appDir) + '"'),
            'set "PYTHONUTF8=1"',
            'set "PYTHONHOME="',
            'set "PYTHONPATH="',
            # uv makes environments without pip, so `edmars setup` installs
            # LSAR's packages with the uv that built this one (edmars.lsar
            # reads EDMARS_UV). That uv may not be on PATH (a private copy
            # in <Dir>\uv), and a user's own uv may be removed later: then
            # the variable is left unset and edmars falls back to PATH and
            # ensurepip instead of a path that no longer exists.
            ('if exist "' + (ConvertTo-CmdPath $uv) + '" set "EDMARS_UV=' + (ConvertTo-CmdPath $uv) + '"'),
            # -P: do not put the current folder first on the import path,
            # so a user's own src\ or edmars\ folder cannot shadow ours.
            ('"' + (ConvertTo-CmdPath $venvPy) + '" -P -m edmars %*'),
            'exit /b %ERRORLEVEL%'
        )
        $cmdText = ($cmdLines -join "`r`n") + "`r`n"
        # cmd.exe reads batch files in the console (OEM) code page.
        $oem = [System.Text.Encoding]::GetEncoding([System.Globalization.CultureInfo]::CurrentCulture.TextInfo.OEMCodePage)
        if ($oem.GetString($oem.GetBytes($cmdText)) -cne $cmdText) {
            Warn "The install path has characters the console code page cannot represent; if 'edmars' does not start, reinstall with -Dir set to a folder with a plain name (for example C:\edm-ars)."
        }
        $launcherTmp = Join-Path $bin ('.edmars-' + $PID + '.tmp')
        [System.IO.File]::WriteAllText($launcherTmp, $cmdText, $oem)
        Move-Item -LiteralPath $launcherTmp -Destination $launcher -Force
        Say "Created $launcher"
        $shLauncher = Join-Path $bin 'edmars'
        if ((Test-Path -LiteralPath $shLauncher) -and -not (Select-String -LiteralPath $shLauncher -SimpleMatch $LauncherMark -Quiet)) {
            Warn "$shLauncher already exists and was not made by this installer, so it was left alone. In Git Bash, type edmars.cmd instead of edmars."
            $shLauncher = $null
        } else {
            $shTmp = Join-Path $bin ('.edmars-sh-' + $PID + '.tmp')
            [System.IO.File]::WriteAllText($shTmp, (Get-ShLauncherText $LauncherMark $ver $appDir $uv $venvPy), (New-Object System.Text.UTF8Encoding $false))
            Move-Item -LiteralPath $shTmp -Destination $shLauncher -Force
            Say "Created $shLauncher (the same command, for Git Bash)"
        }

        # ---- 7. PATH --------------------------------------------------------------------
        Step '7/8' 'PATH'
        $pathModified = $false
        $pathEntry = $null
        $sessionPath = ($env:Path -split ';') | Where-Object { $_ } | ForEach-Object { [Environment]::ExpandEnvironmentVariables($_).TrimEnd('\') }
        $onSessionPath = @($sessionPath | Where-Object { $_ -ieq $bin.TrimEnd('\') }).Count -gt 0
        if ($NoModifyPath) {
            Say 'Left unchanged (-NoModifyPath).'
        } else {
            # Read the raw user PATH from the registry: going through
            # [Environment] would expand %VARIABLES% in it and write them
            # back expanded, silently changing the user's other entries.
            $key = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Environment')
            try {
                $raw = [string]$key.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
                $kind = [Microsoft.Win32.RegistryValueKind]::ExpandString
                if (($key.GetValueNames() -contains 'Path') -and ($key.GetValueKind('Path') -eq [Microsoft.Win32.RegistryValueKind]::String)) {
                    $kind = [Microsoft.Win32.RegistryValueKind]::String
                }
                $newRaw = Merge-UserPath $raw $bin ($kind -eq [Microsoft.Win32.RegistryValueKind]::ExpandString)
                if ($null -eq $newRaw) {
                    Say "$bin is already on your user PATH."
                } else {
                    $key.SetValue('Path', $newRaw, $kind)
                    $pathEntry = ($newRaw -split ';')[-1]
                    $pathModified = $true
                    Say "Added $bin to your user PATH."
                }
            } finally {
                $key.Close()
            }
            if ($pathModified) {
                # Tell Explorer (and so new terminals) that the environment
                # changed: setting a user variable through .NET broadcasts
                # WM_SETTINGCHANGE.
                [Environment]::SetEnvironmentVariable('EDMARS_INSTALLER_BROADCAST', '1', 'User')
                [Environment]::SetEnvironmentVariable('EDMARS_INSTALLER_BROADCAST', $null, 'User')
            }
            if (-not $onSessionPath) {
                # Make `edmars` work in this window right away.
                $env:Path = $env:Path.TrimEnd(';') + ';' + $bin
                $onSessionPath = $true
            }
        }

        # ---- record the install -----------------------------------------------------------
        $versions = @()
        if (Test-Path -LiteralPath $versionsFile) {
            $versions = @(Get-Content -LiteralPath $versionsFile | Where-Object { $_ -and ($_ -ne $ver) } | ForEach-Object { [string]$_ })
        }
        $versions += $ver
        # Keep the current and the previous version (a study started before
        # this update may still be running from it); remove older ones. Only
        # folders this installer recorded are ever removed.
        $keep = 2
        if ($versions.Count -gt $keep) {
            foreach ($old in $versions[0..($versions.Count - $keep - 1)]) {
                if ($old -notmatch '^[0-9A-Za-z][0-9A-Za-z.+-]*$') { continue }
                foreach ($p in @((Join-Path $appParent $old), (Join-Path $base "venv-$old"))) {
                    if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue }
                }
            }
            $versions = $versions[($versions.Count - $keep)..($versions.Count - 1)]
        }
        $utf8 = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllText($versionsFile, (($versions -join "`n") + "`n"), $utf8)
        $prevOut = $null
        if ($previous -and ($previous -ne $ver)) { $prevOut = $previous }
        $manifest = [ordered]@{
            schema           = 1
            installer        = 'install.ps1'
            version          = $ver
            previous_version = $prevOut
            installed_at     = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
            source           = $sourceKind
            install_dir      = $base
            app_root         = $appDir
            venv             = $venv
            python           = $venvPy
            pth              = $pth
            uv               = $uv
            uv_private       = $uvPrivate
            bin_dir          = $bin
            launcher         = $launcher
            sh_launcher      = $shLauncher
            path_modified    = $pathModified
            path_entry       = $pathEntry
        }
        [System.IO.File]::WriteAllText((Join-Path $base 'install.json'), ($manifest | ConvertTo-Json), $utf8)

        # ---- 8. smoke test and setup ----------------------------------------------------------
        Step '8/8' 'Checking the edmars command'
        Invoke-Checked "'edmars version'" $launcher @('version')

        Say ''
        Say "EDM-ARS $ver is installed."
        if ($onSessionPath) { $runCmd = 'edmars' } else { $runCmd = '"' + $launcher + '"' }
        if ($NoOnboard) {
            Say "Next, set it up:  $runCmd setup"
        } elseif ($interactive) {
            Say "Starting the setup wizard. You can leave it at any time and run 'edmars setup' later."
            $saved = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            try { & $launcher setup } finally { $ErrorActionPreference = $saved }
            if ($LASTEXITCODE -ne 0) { Warn "Setup did not finish. Run 'edmars setup' any time to continue." }
        } else {
            Say "No console is available for the setup wizard. Run it yourself:  $runCmd setup"
        }
        Say ''
        Say "To use EDM-ARS, type:  $runCmd"
        if ($pathModified) { Say '(Other terminal windows that were already open need to be reopened first.)' }
    } finally {
        if ($pushed) { Pop-Location }
        foreach ($name in $savedEnv.Keys) { [Environment]::SetEnvironmentVariable($name, $savedEnv[$name], 'Process') }
        if ($tmp -and (Test-Path -LiteralPath $tmp)) { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue }
        if ($stage -and (Test-Path -LiteralPath $stage)) { Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue }
    }
} @args

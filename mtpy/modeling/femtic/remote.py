""" Remote Slurm-cluster helpers for the FEMTIC workflow.

You might want to exclude most, if not all, of this. This might not be a 
reasonable add to mtpy-v2. its relevance to the actual  topic is pretty 
peripheral, but it can be extremely convenient. Perhaps I'll roll it into 
my own separate repo.

Built on Paramiko SSH/SFTP. :class:`RemoteCluster` owns a single SSH
connection (and works as a context manager) and exposes:

* remote command execution (``run``);
* Slurm job submission (``submit_sbatch``) and queue polling (``squeue`` /
  ``squeue_rows``);
* recursive directory upload (``send_dir``) and download (``get_dir``) over
  SFTP, plus a recursive remote glob (``glob``).

The pure stdout parsers (:func:`parse_sbatch_job_id`,
:func:`parse_squeue_stdout`, :func:`columns_from_squeue_format`) are kept as
module-level functions since they need no connection.

paramiko is technically an optional dependency: the module imports without it, 
and :class:`RemoteCluster` raises a clear :class:`ImportError` on construction
if it is missing.

@author: oaazeved

"""

from __future__ import annotations

import os
import re
import stat
import fnmatch
import posixpath
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Iterable, List, Tuple, Dict

from loguru import logger

try:
    import paramiko
    _HAVE_PARAMIKO = True
except ImportError:  # pragma: no cover
    paramiko = None
    _HAVE_PARAMIKO = False


__all__ = [
    "RemoteCluster", "RemoteCommandResult", "SqueueResult",
    "parse_sbatch_job_id", "parse_squeue_stdout", "columns_from_squeue_format",
]


def _require_paramiko():
    """Return the paramiko module, or raise a clear error if it is absent.

    :raises ImportError: If paramiko is not installed.
    """
    if not _HAVE_PARAMIKO:
        raise ImportError(
            "paramiko is required for femtic_remote.RemoteCluster but is not "
            "installed in the current environment. Install it "
            "(e.g. `pip install paramiko`) to use the SSH/SFTP cluster "
            "helpers. The stdout parsers in this module work without it."
        )
    return paramiko



# Result container classes; not super necessary, but sort of nice to have

@dataclass(frozen=True)
class RemoteCommandResult:
    """Result of a remote command: the full command, exit status, and output."""
    command: str
    exit_status: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class SqueueResult:
    """Result of an ``squeue`` poll: the command, exit status, and raw output."""
    command: str
    exit_status: int
    stdout: str
    stderr: str



# stdout parsers (no connection needed)

_SBATCH_JOBID_RE = re.compile(r"Submitted batch job\s+(\d+)")


def parse_sbatch_job_id(sbatch_stdout: str) -> Optional[int]:
    """Extract the job id from ``Submitted batch job <id>`` output.

    :param sbatch_stdout: Raw stdout from ``sbatch``.
    :type sbatch_stdout: str
    :return: The integer job id, or ``None`` if not found.
    :rtype: int or None
    """
    m = _SBATCH_JOBID_RE.search(sbatch_stdout or "")
    return int(m.group(1)) if m else None


def parse_squeue_stdout(stdout: str, *, delimiter: Optional[str] = None,
                        columns: Optional[Iterable[str]] = None,
                        strip: bool = True,
                        skip_empty: bool = True) -> List[Dict[str, str]]:
    """Parse ``squeue`` stdout into a list of per-job dictionaries.

    :param stdout: Raw stdout returned by squeue.
    :type stdout: str
    :param delimiter: Field delimiter; ``None`` splits on whitespace.
    :type delimiter: str or None, optional
    :param columns: Column names in order; ``None`` uses ``col0, col1, ...``.
    :type columns: iterable of str or None, optional
    :param strip: Strip whitespace around fields, defaults to ``True``.
    :type strip: bool, optional
    :param skip_empty: Skip blank lines, defaults to ``True``.
    :type skip_empty: bool, optional
    :return: One dict per job row.
    :rtype: list of dict
    :raises ValueError: If a row's field count differs from ``columns``.
    """
    rows: List[Dict[str, str]] = []
    if not stdout:
        return rows
    columns_list = list(columns) if columns is not None else None
    for line in stdout.splitlines():
        if skip_empty and not line.strip():
            continue
        parts = line.split(delimiter) if delimiter else line.split()
        if strip:
            parts = [p.strip() for p in parts]
        if columns_list is not None:
            if len(parts) != len(columns_list):
                raise ValueError(
                    f"squeue parse error: expected {len(columns_list)} fields, "
                    f"got {len(parts)} for line:\n{line}")
            rows.append(dict(zip(columns_list, parts)))
        else:
            rows.append({f"col{i}": v for i, v in enumerate(parts)})
    return rows


def columns_from_squeue_format(format_str: str) -> List[str]:
    """Infer column names from an ``squeue --format`` string.

    :param format_str: e.g. ``"%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"``.
    :type format_str: str
    :return: Column names, e.g. ``['JOBID', 'PARTITION', ...]``.
    :rtype: list of str
    """
    mapping = {
        "i": "JOBID", "P": "PARTITION", "j": "NAME", "u": "USER",
        "t": "STATE", "M": "TIME", "D": "NODES", "R": "NODELIST",
        "l": "TIMELIMIT", "S": "START_TIME", "T": "STATE_LONG", "C": "CPUS",
    }
    cols = []
    for tok in format_str.split():
        key = tok.lstrip("%0123456789.")
        cols.append(mapping.get(key, key.upper()))
    return cols



# RemoteCluster


class RemoteCluster:
    """An SSH/SFTP session to a remote Slurm cluster.

    Owns one :class:`paramiko.SSHClient`. Use as a context manager so the
    connection is always closed::

        with RemoteCluster(host, user, pw, host_key_policy="reject") as c:
            c.send_dir(local_dir, remote_dir, debug=True)
            job_id, _ = c.submit_sbatch(remote_dir, "femtic_run.sbatch")
            print(c.squeue_rows())

    :param hostname: Hostname or IP of the SSH server.
    :type hostname: str
    :param username: SSH username.
    :type username: str
    :param password: SSH password.
    :type password: str
    :param port: SSH port, defaults to ``22``.
    :type port: int, optional
    :param known_hosts: Optional known_hosts file to preload; ``None`` loads
        the system host keys.
    :type known_hosts: str or None, optional
    :param host_key_policy: ``"auto_add"`` or ``"reject"``, defaults to
        ``"auto_add"``.
    :type host_key_policy: str, optional
    :param timeout: Connect timeout in seconds, defaults to ``20``.
    :type timeout: int, optional
    :param debug: Default debug-print flag for all methods, defaults to
        ``False``.
    :type debug: bool, optional
    :param connect: Connect immediately on construction, defaults to ``True``.
    :type connect: bool, optional
    :raises ImportError: If paramiko is not installed.
    :raises ValueError: If ``host_key_policy`` is invalid.
    """

    def __init__(self, hostname: str, username: str, password: str,
                 port: int = 22, known_hosts: Optional[str] = None,
                 host_key_policy: str = "auto_add", timeout: int = 20,
                 debug: bool = False, connect: bool = True):
        _require_paramiko()
        if host_key_policy not in ("auto_add", "reject"):
            raise ValueError("host_key_policy must be 'auto_add' or 'reject'")
        self.hostname = hostname
        self.username = username
        self._password = password
        self.port = port
        self.known_hosts = known_hosts
        self.host_key_policy = host_key_policy
        self.timeout = timeout
        self.debug = debug
        self._client = None
        self.logger = logger
        if connect:
            self.connect()

    # lifecycle 
    def connect(self) -> "RemoteCluster":
        """Open the SSH connection (no-op if already connected).

        :return: ``self`` (for chaining).
        :rtype: RemoteCluster
        """
        if self._client is not None:
            return self
        client = paramiko.SSHClient()
        if self.known_hosts:
            client.load_host_keys(self.known_hosts)
            self._dbg(f"Loaded known_hosts from {self.known_hosts}")
        else:
            client.load_system_host_keys()
            self._dbg("Loaded system known_hosts")
        if self.host_key_policy == "reject":
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
        else:
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._dbg(f"Connecting to {self.hostname}:{self.port} as {self.username}")
        client.connect(hostname=self.hostname, port=self.port,
                       username=self.username, password=self._password,
                       look_for_keys=False, allow_agent=False,
                       timeout=self.timeout)
        self._dbg("SSH connection established")
        self._client = client
        return self

    def close(self) -> None:
        """Close the SSH connection (safe to call multiple times)."""
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None
                self._dbg("SSH connection closed")

    def __enter__(self) -> "RemoteCluster":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def client(self):
        """The live :class:`paramiko.SSHClient`.

        :raises RuntimeError: If not connected.
        """
        if self._client is None:
            raise RuntimeError("RemoteCluster is not connected; call connect().")
        return self._client

    @property
    def connected(self) -> bool:
        """Whether the SSH connection is currently open."""
        return self._client is not None

    # internals 
    def _dbg(self, msg: str, debug: Optional[bool] = None) -> None:
        if self.debug if debug is None else debug:
            self.logger.debug(msg)

    # command execution ------------------------------------------------
    def run(self, command: str, workdir: Optional[str] = None,
            env: Optional[Dict[str, str]] = None, get_pty: bool = False,
            timeout: Optional[float] = None,
            debug: Optional[bool] = None) -> RemoteCommandResult:
        """Run a command on the remote host through a login shell.

        :param command: The command to run.
        :type command: str
        :param workdir: Optional remote working directory (``cd`` first).
        :type workdir: str or None, optional
        :param env: Optional environment variables to prefix.
        :type env: dict or None, optional
        :param get_pty: Request a pseudo-terminal, defaults to ``False``.
        :type get_pty: bool, optional
        :param timeout: Command timeout in seconds.
        :type timeout: float or None, optional
        :param debug: Override the instance debug flag.
        :type debug: bool or None, optional
        :return: The command result.
        :rtype: RemoteCommandResult
        """
        env_prefix = ""
        if env:
            def q(v: str) -> str:
                return "'" + v.replace("'", "'\"'\"'") + "'"
            env_prefix = " ".join([f"{k}={q(v)}" for k, v in env.items()]) + " "
        cd_prefix = f"cd {shlex.quote(workdir)} && " if workdir else ""
        full = f"bash -lc {repr(env_prefix + cd_prefix + command)}"
        self._dbg(f"Remote exec: {full}", debug)
        stdin, stdout, stderr = self.client.exec_command(
            full, get_pty=get_pty, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        exit_status = stdout.channel.recv_exit_status()
        self._dbg(f"Exit status: {exit_status}", debug)
        if (self.debug if debug is None else debug) and out.strip():
            self._dbg(f"STDOUT:\n{out}", debug)
        if (self.debug if debug is None else debug) and err.strip():
            self._dbg(f"STDERR:\n{err}", debug)
        return RemoteCommandResult(command=full, exit_status=exit_status,
                                   stdout=out, stderr=err)

    # Slurm ----
    def submit_sbatch(self, remote_workdir: str, sbatch_script: str,
                      sbatch_args: Optional[Sequence[str]] = None,
                      ensure_remote_dir: bool = False,
                      debug: Optional[bool] = None
                      ) -> Tuple[int, RemoteCommandResult]:
        """Submit a Slurm job (``cd <workdir> && sbatch <args> <script>``).

        :param remote_workdir: Directory to submit from.
        :type remote_workdir: str
        :param sbatch_script: The sbatch script filename.
        :type sbatch_script: str
        :param sbatch_args: Extra args to pass to ``sbatch``.
        :type sbatch_args: sequence of str or None, optional
        :param ensure_remote_dir: ``mkdir -p`` the workdir first, defaults to
            ``False``.
        :type ensure_remote_dir: bool, optional
        :param debug: Override the instance debug flag.
        :type debug: bool or None, optional
        :return: ``(job_id, result)``.
        :rtype: tuple(int, RemoteCommandResult)
        :raises RuntimeError: If submission fails or the id can't be parsed.
        """
        if ensure_remote_dir:
            res = self.run(f"mkdir -p '{remote_workdir}'", debug=debug)
            if res.exit_status != 0:
                raise RuntimeError(
                    f"Failed to create remote_workdir:\n{res.stderr}")
        args = " ".join(sbatch_args or [])
        cmd = f"sbatch {args} '{sbatch_script}'".strip()
        result = self.run(cmd, workdir=remote_workdir, debug=debug)
        if result.exit_status != 0:
            raise RuntimeError(
                "sbatch failed\n"
                f"Exit status: {result.exit_status}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}")
        job_id = parse_sbatch_job_id(result.stdout)
        if job_id is None:
            s = (result.stdout or "").strip()
            if s.isdigit():
                job_id = int(s)
            else:
                raise RuntimeError(
                    "sbatch succeeded but job id could not be parsed.\n"
                    f"STDOUT:\n{result.stdout}\n"
                    f"STDERR:\n{result.stderr}")
        self._dbg(f"Submitted Slurm job id: {job_id}", debug)
        return job_id, result

    def squeue(self, user: Optional[str] = "--me", job_id: Optional[int] = None,
               partition: Optional[str] = None,
               states: Optional[Sequence[str]] = None,
               format_str: str = "%.18i %.9u %.8T %.10M %.6D %.30j",
               sort: str = "i", noheader: bool = True,
               delimiter: Optional[str] = None, workdir: Optional[str] = None,
               timeout: Optional[float] = 20.0,
               debug: Optional[bool] = None) -> SqueueResult:
        """Poll ``squeue`` once. Does not parse the output (see
        :meth:`squeue_rows`).

        :param user: ``-u`` filter (``"--me"`` for the current user).
        :param job_id: ``-j`` filter.
        :param partition: ``-p`` filter.
        :param states: ``-t`` state filter.
        :param format_str: ``-o`` format string.
        :param noheader: Add ``--noheader``, defaults to ``True``.
        :param delimiter: If set, replaces spaces in ``format_str``.
        :param workdir: Optional remote working directory.
        :param timeout: Command timeout in seconds.
        :param debug: Override the instance debug flag.
        :return: The raw squeue result.
        :rtype: SqueueResult
        """
        args = ["squeue"]
        if user is not None:
            args += ["-u", shlex.quote(user)]
        if job_id is not None:
            args += ["-j", str(int(job_id))]
        if partition is not None:
            args += ["-p", shlex.quote(partition)]
        if states is not None:
            args += ["-t", shlex.quote(",".join(states))]
        if noheader:
            args.append("--noheader")
        if delimiter is not None:
            format_str = format_str.replace(" ", delimiter)
        args += ["-o", shlex.quote(format_str)]
        cmd = " ".join(args)
        if workdir is not None:
            cmd = f"cd {shlex.quote(workdir)} && {cmd}"
        full = f"bash -lc {shlex.quote(cmd)}"
        self._dbg(f"Remote exec: {full}", debug)
        stdin, stdout, stderr = self.client.exec_command(
            full, get_pty=False, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        exit_status = stdout.channel.recv_exit_status()
        self._dbg(f"Exit status: {exit_status}", debug)
        if (self.debug if debug is None else debug) and out.strip():
            self._dbg(f"STDOUT:\n{out}", debug)
        if (self.debug if debug is None else debug) and err.strip():
            self._dbg(f"STDERR:\n{err}", debug)
        return SqueueResult(command=full, exit_status=exit_status,
                            stdout=out, stderr=err)

    def squeue_rows(self, columns: Optional[Iterable[str]] = None,
                    format_str: str = "%.18i %.9u %.8T %.10M %.6D %.30j",
                    delimiter: Optional[str] = None,
                    **squeue_kwargs) -> List[Dict[str, str]]:
        """Poll ``squeue`` and parse the result into a list of dicts.

        Column names are inferred from ``format_str`` when ``columns`` is not
        given.

        :param columns: Explicit column names, or ``None`` to infer them.
        :param format_str: ``-o`` format string (also used to infer columns).
        :param delimiter: Optional field delimiter.
        :param squeue_kwargs: Forwarded to :meth:`squeue`.
        :return: One dict per job row.
        :rtype: list of dict
        """
        if columns is None:
            columns = columns_from_squeue_format(format_str)
        result = self.squeue(format_str=format_str, delimiter=delimiter,
                             **squeue_kwargs)
        return parse_squeue_stdout(result.stdout, delimiter=delimiter,
                                   columns=columns)

    # SFTP transfers ---------------------------------------------------
    @staticmethod
    def _sftp_mkdir_p(sftp, remote_path: str, debug: bool = False) -> None:
        """Recursively create remote directories (``mkdir -p`` behaviour)."""
        remote_path = posixpath.normpath(remote_path)
        if remote_path in (".", "/", ""):
            return
        cur = ""
        for p in remote_path.strip("/").split("/"):
            cur = "/" + p if not cur else cur + "/" + p
            try:
                sftp.stat(cur)
            except IOError:
                if debug:
                    logger.debug(f"Creating remote directory: {cur}")
                sftp.mkdir(cur)

    def send_dir(self, local_dir, remote_dir: str, preserve_mtime: bool = True,
                 debug: Optional[bool] = None) -> None:
        """Recursively upload a local directory over SFTP (existing connection).

        :param local_dir: Local directory to upload.
        :type local_dir: str or os.PathLike
        :param remote_dir: Destination directory on the remote host.
        :type remote_dir: str
        :param preserve_mtime: Preserve file mtimes, defaults to ``True``.
        :type preserve_mtime: bool, optional
        :param debug: Override the instance debug flag.
        :type debug: bool or None, optional
        :raises ValueError: If ``local_dir`` is not a directory.
        """
        local_dir = Path(local_dir).expanduser().resolve()
        if not local_dir.is_dir():
            raise ValueError(f"local_dir is not a directory: {local_dir}")
        remote_dir = posixpath.normpath(remote_dir)
        dbg = self.debug if debug is None else debug
        self._dbg(f"Local directory: {local_dir}", debug)
        self._dbg(f"Remote directory: {remote_dir}", debug)
        sftp = self.client.open_sftp()
        try:
            self._sftp_mkdir_p(sftp, remote_dir, debug=dbg)
            for root, dirs, files in os.walk(local_dir):
                root_path = Path(root)
                rel = root_path.relative_to(local_dir).as_posix()
                remote_root = (remote_dir if rel == "."
                               else posixpath.join(remote_dir, rel))
                self._sftp_mkdir_p(sftp, remote_root, debug=dbg)
                for d in dirs:
                    self._sftp_mkdir_p(sftp, posixpath.join(remote_root, d),
                                       debug=dbg)
                for f in files:
                    local_file = root_path / f
                    remote_file = posixpath.join(remote_root, f)
                    self._dbg(f"Uploading file: {local_file} -> {remote_file}",
                              debug)
                    sftp.put(str(local_file), remote_file)
                    if preserve_mtime:
                        st = local_file.stat()
                        sftp.utime(remote_file, (st.st_atime, st.st_mtime))
            self._dbg("SFTP upload complete", debug)
        finally:
            sftp.close()

    def get_dir(self, remote_dir: str, local_dir: str,
                include_globs: Optional[Sequence[str]] = None,
                exclude_globs: Optional[Sequence[str]] = None
                ) -> Dict[str, str]:
        """Recursively download a remote directory over SFTP with glob filters.

        ``exclude_globs`` applies to both files and directories (matched
        against the path relative to ``remote_dir``); ``include_globs`` applies
        to files only, and a pattern with no glob metacharacters is treated as
        a substring (``*pat*``).

        :param remote_dir: Remote source directory.
        :type remote_dir: str
        :param local_dir: Local destination directory.
        :type local_dir: str
        :param include_globs: File-only include patterns.
        :type include_globs: sequence of str or None, optional
        :param exclude_globs: Path exclude patterns (files and dirs).
        :type exclude_globs: sequence of str or None, optional
        :return: Mapping of downloaded ``remote_path -> local_path``.
        :rtype: dict
        """
        exclude_globs = list(exclude_globs or [])
        include_globs = list(include_globs or [])

        def _is_excluded(rel_path: str) -> bool:
            rel_path = rel_path.lstrip("/")
            return any(fnmatch.fnmatch(rel_path, p) for p in exclude_globs)

        def _norm(p: str) -> str:
            return p if any(ch in p for ch in "*?[]") else f"*{p}*"

        include_patterns = [_norm(p) for p in include_globs]

        def _is_included_file(rel_path: str) -> bool:
            if not include_patterns:
                return True
            rel_path = rel_path.lstrip("/")
            return any(fnmatch.fnmatch(rel_path, p) for p in include_patterns)

        remote_dir = posixpath.normpath(remote_dir)
        local_dir = os.path.normpath(local_dir)
        os.makedirs(local_dir, exist_ok=True)
        downloaded: Dict[str, str] = {}
        sftp = self.client.open_sftp()

        def _recursive_get(remote_path, local_path, rel_root):
            try:
                entries = sftp.listdir_attr(remote_path)
            except IOError as exc:
                raise RuntimeError(
                    f"Failed to list directory: {remote_path}") from exc
            for entry in entries:
                remote_item = posixpath.join(remote_path, entry.filename)
                local_item = os.path.join(local_path, entry.filename)
                rel_path = posixpath.join(rel_root, entry.filename).lstrip("/")
                if _is_excluded(rel_path):
                    continue
                if stat.S_ISDIR(entry.st_mode):
                    os.makedirs(local_item, exist_ok=True)
                    _recursive_get(remote_item, local_item, rel_path)
                else:
                    if not _is_included_file(rel_path):
                        continue
                    sftp.get(remote_item, local_item)
                    downloaded[remote_item] = local_item

        try:
            _recursive_get(remote_dir, local_dir, "")
        finally:
            try:
                sftp.close()
            except Exception:
                pass
        return downloaded

    def glob(self, base_dir: str, name_glob: str) -> List[str]:
        """Recursively list remote files whose basename matches ``name_glob``.

        :param base_dir: Remote directory to walk.
        :type base_dir: str
        :param name_glob: Filename glob (e.g. ``"*.cnv"``).
        :type name_glob: str
        :return: Matching remote file paths.
        :rtype: list of str
        """
        sftp = self.client.open_sftp()
        results: List[str] = []
        stack = [base_dir]
        try:
            while stack:
                cur = stack.pop()
                for entry in sftp.listdir_attr(cur):
                    full = posixpath.join(cur, entry.filename)
                    if stat.S_ISDIR(entry.st_mode):
                        stack.append(full)
                    elif stat.S_ISREG(entry.st_mode) and \
                            fnmatch.fnmatch(entry.filename, name_glob):
                        results.append(full)
        finally:
            sftp.close()
        return results

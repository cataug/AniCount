#!/usr/bin/env python3
"""Download AniCount model weights and published aerial-wildlife datasets.

Python standard library + curl only. No installation, extraction, or GPU use.
Default: model weights. --all adds Delplanque, Koger and OWL/caribou data.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode


@dataclass
class Download:
    url: str
    path: str
    size: int = 0
    checksum: str = ""
    estimate: int = 0


def curl_command():
    # No internal retries for payloads: curl rolls back output before retrying.
    # A fresh process must read the current partial length on every attempt.
    return ["curl", "--disable", "--ipv4", "--http1.1", "--location", "--fail",
            "--show-error", "--connect-timeout", "20", "--retry", "0"]


def get_json(url):
    result = subprocess.run(curl_command() + ["--silent", "--retry", "1", "--max-time", "45", url],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Metadata HTTP request failed ({result.returncode}): {result.stderr.strip()[-600:]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Repository returned non-JSON content (possibly a browser access check)") from error


def file_hash(path, algorithm):
    algorithm = algorithm.lower().replace("-", "")
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_file(path, job):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if job.size and path.stat().st_size != job.size:
        return False
    if job.checksum:
        algorithm, expected = job.checksum.split(":", 1)
        return file_hash(path, algorithm) == expected.lower()
    return True


def transfer_payload(job, partial, idle_retries=8, stall_seconds=180, retry_delay=3):
    """Resume after each failed request, preserving all newly received bytes.

    The retry budget counts consecutive requests without progress. Requests
    that advance the saved prefix do not exhaust that budget on a huge file.
    """
    idle_failures = 0
    attempt = 0
    transient_codes = {5, 6, 7, 16, 18, 28, 35, 52, 55, 56, 92, 95}
    transient_http = {408, 425, 429, 500, 502, 503, 504}
    while True:
        before = partial.stat().st_size if partial.is_file() else 0
        attempt += 1
        print(f"[RESUME] Attempt {attempt}; saved {before:,} bytes "
              f"({before/1024**3:.3f} GiB)", flush=True)
        command = curl_command() + [
            "--speed-limit", "1024", "--speed-time", str(stall_seconds),
            "--continue-at", str(before), "--output", str(partial),
            "--write-out", "%{http_code}", job.url,
        ]
        # curl progress remains live on stderr; stdout is only the HTTP code.
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
        status = (result.stdout or "").strip()
        http_code = int(status[-3:]) if status[-3:].isdigit() else 0
        after = partial.stat().st_size if partial.is_file() else 0
        if after < before:
            raise RuntimeError("Partial file shrank unexpectedly; stopped to protect existing data")
        if result.returncode == 0:
            if after <= before and not (job.size and after == job.size):
                raise RuntimeError("Request returned success without new data; partial file retained")
            return
        # A server may reject a request past EOF if the previous request already
        # delivered the full object but failed while closing the connection.
        if (job.size and after == job.size) or http_code == 416:
            if (job.checksum or job.size) and valid_file(partial, job):
                return
            if http_code == 416:
                raise RuntimeError("HTTP 416: saved length/checksum does not match the remote object; partial retained")
        if result.returncode in {33, 36}:
            raise RuntimeError("Host refused byte-range resume; existing partial retained (no automatic restart)")
        retryable = result.returncode in transient_codes or (
            result.returncode == 22 and http_code in transient_http)
        if not retryable:
            raise RuntimeError(f"curl failed ({result.returncode}, HTTP {http_code}); partial retained")
        advanced = after - before
        idle_failures = 0 if advanced else idle_failures + 1
        print(f"[RETRY] curl={result.returncode}, HTTP={http_code}; "
              f"kept {after:,} bytes (+{advanced:,}); "
              f"no-progress failures={idle_failures}/{idle_retries + 1}", flush=True)
        if idle_failures > idle_retries:
            raise RuntimeError("Repeated requests made no progress; partial retained for a later run")
        time.sleep(min(30, retry_delay * max(1, idle_failures)))


def download(job, idle_retries=8, stall_seconds=180):
    target = Path(job.path)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[FILE] {target}", flush=True)
    if valid_file(target, job):
        print("[OK] Already downloaded", flush=True)
        return
    if target.exists():
        backup = target.with_name(target.name + f".invalid.{time.time_ns()}")
        target.rename(backup)
        print(f"[BACKUP] Previous invalid file: {backup.name}", flush=True)
    partial = target.with_name(target.name + ".part")
    if (job.checksum or job.size) and valid_file(partial, job):
        partial.replace(target)
        print("[OK] Completed partial file verified", flush=True)
        return
    transfer_payload(job, partial, idle_retries, stall_seconds)
    print("[VERIFY] Checking downloaded file", flush=True)
    if not valid_file(partial, job):
        invalid = partial.with_name(partial.name + f".invalid.{time.time_ns()}")
        partial.rename(invalid)
        raise RuntimeError(f"Size/checksum mismatch; retained as {invalid.name}")
    partial.replace(target)
    if not job.checksum:
        # This is a local integrity record, not a publisher-provided checksum.
        target.with_name(target.name + ".sha256.local").write_text(
            file_hash(target, "sha256") + "  " + target.name + "\n")
    print("[OK] Downloaded", flush=True)


def model_jobs(root):
    zenodo = "https://zenodo.org/records/20802844/files/"
    return [
        Download(
            "https://github.com/ConservationDronesAI/DroneMegaDetector/releases/download/"
            "v1.1/DroneMegaDetector_RFDETRMedium_2026.10.04.pth",
            str(root / "weights/DroneMegaDetector_RFDETRMedium_2026.10.04.pth"),
            estimate=1024**3),
        Download(zenodo + "OWL-C.pth?download=1", str(root / "weights/OWL-C.pth"),
                 checksum="md5:4d1cb40b4ee16d531ded09c16333d73b", estimate=250*1024**2),
        Download(zenodo + "OWL-D.pth?download=1", str(root / "weights/OWL-D.pth"),
                 checksum="md5:30ff57bba505de4be743c2e0ef0c23ed", estimate=4*1024**3),
    ]


def dataverse_jobs(root, host, doi, name):
    print(f"[METADATA] {name}: {doi}", flush=True)
    query = urlencode({"persistentId": "doi:" + doi})
    payload = get_json(f"{host}/api/datasets/:persistentId/?{query}")
    if payload.get("status") != "OK":
        raise RuntimeError("Dataverse did not return an OK dataset response")
    dataset = payload["data"]
    version = dataset["latestVersion"]
    if version.get("versionState") != "RELEASED":
        raise RuntimeError("Dataset response is not a published version")
    directory = root / "DATA" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "dataset_metadata.json").write_text(json.dumps(payload, indent=2))
    jobs = []
    for entry in version.get("files", []):
        if entry.get("restricted"):
            raise RuntimeError("Dataset includes restricted files; automatic complete download unavailable")
        data = entry["dataFile"]
        relative = PurePosixPath(entry.get("directoryLabel") or "")
        filename = PurePosixPath(data["filename"])
        if relative.is_absolute() or ".." in relative.parts or len(filename.parts) != 1:
            raise RuntimeError("Unsafe filename in repository metadata")
        checksum = data.get("checksum") or {}
        value = ""
        if checksum.get("type") and checksum.get("value"):
            value = checksum["type"].lower().replace("-", "") + ":" + checksum["value"]
        # An ingested tabular file's checksum describes Dataverse's TSV, not
        # the original file we request. Do not incorrectly verify that TSV hash.
        size = int(data.get("filesize", 0))
        if data.get("originalFileFormat"):
            size = int(data.get("originalFileSize") or 0)
            value = ""
        # Keep original tabular content rather than an automatic TSV conversion.
        jobs.append(Download(
            f"{host}/api/access/datafile/{data['id']}?format=original",
            str(directory.joinpath(*relative.parts, filename.name)),
            size=size, checksum=value))
    if not jobs:
        raise RuntimeError("No published files found")
    return jobs


def dataset_jobs(root, errors):
    zenodo = "https://zenodo.org/records/20802844/files/"
    jobs = [
        Download(zenodo + "train.zip?download=1", str(root / "DATA/OWL_Caribou/train.zip"),
                 checksum="md5:d55575fecc0e5e6d7180c96b74e75097", estimate=12*1024**3),
        Download(zenodo + "test.zip?download=1", str(root / "DATA/OWL_Caribou/test.zip"),
                 checksum="md5:086a0430c769b164cbcc9d2d48db6fac", estimate=1400*1024**2),
    ]
    sources = [
        ("https://dataverse.uliege.be", "10.58119/ULG/MIRUU5", "Delplanque_African_Mammals"),
        ("https://edmond.mpg.de", "10.17617/3.EMRZGH", "Koger_African_Drones"),
    ]
    for host, doi, name in sources:
        try:
            jobs.extend(dataverse_jobs(root, host, doi, name))
        except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
            errors.append({"source": name, "error": str(error), "doi": "https://doi.org/" + doi})
            print(f"[SOURCE FAILED] {name}: {error}", flush=True)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Weights and all three datasets")
    mode.add_argument("--datasets", action="store_true", help="Datasets only")
    mode.add_argument("--models", action="store_true", help="Weights only (default)")
    mode.add_argument("--resume", action="store_true", help="Use saved manifest, without repository metadata requests")
    parser.add_argument("--list", action="store_true", help="Build manifest and show size; do not download")
    parser.add_argument("--show-report", action="store_true", help="Print errors from the previous run and stop")
    parser.add_argument("--only", default="", help="Limit paths to this case-insensitive substring, e.g. Koger")
    parser.add_argument("--idle-retries", type=int, default=8, help="Retries after consecutive requests without progress")
    parser.add_argument("--stall-seconds", type=int, default=180, help="Reconnect after this many seconds below 1 KiB/s")
    parser.add_argument("--root", type=Path, default=Path("/home/tahiti/AniCount"))
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.show_report:
        report_path = root / "download_report.json"
        report = json.loads(report_path.read_text())
        print(f"[REPORT] {report_path}", flush=True)
        for entry in report.get("errors", []):
            print(json.dumps(entry, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.idle_retries < 0 or args.stall_seconds < 1:
        parser.error("--idle-retries must be >= 0 and --stall-seconds must be >= 1")
    if not shutil.which("curl"):
        raise RuntimeError("curl is required")
    # Held until process termination/return; a second downloader must not write
    # to the same .part files concurrently.
    lock = (root / ".anicount_download.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError("Another AniCount downloader is running for this directory") from error
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()) + "\n")
    lock.flush()
    manifest = root / "download_manifest.json"
    if args.resume:
        cached = json.loads(manifest.read_text())
        jobs = [Download(**item) for item in cached["files"]]
        errors = list(cached.get("source_errors", []))
        for job in jobs:
            if not Path(job.path).resolve().is_relative_to(root):
                raise RuntimeError("Saved manifest contains a path outside --root")
        print("[RESUME] Loaded existing manifest; no repository metadata requests", flush=True)
        for error in errors:
            print(f"[UNRESOLVED SOURCE] {error.get('source', '?')}: "
                  f"{error.get('error', '?')} (use --all later to refresh metadata)", flush=True)
    else:
        errors = []
        jobs = [] if args.datasets else model_jobs(root)
        if args.all or args.datasets:
            jobs.extend(dataset_jobs(root, errors))
        manifest.write_text(json.dumps({"files": [asdict(job) for job in jobs],
                                        "source_errors": errors}, indent=2))
    if args.only:
        jobs = [job for job in jobs if args.only.lower() in job.path.lower()]
    if not jobs:
        raise RuntimeError("No matching files in the download plan")
    remaining = 0
    for job in jobs:
        path = Path(job.path)
        if path.is_file():
            print(f"[CHECK] {path}", flush=True)
        if valid_file(path, job):
            continue
        partial = path.with_name(path.name + ".part")
        have = partial.stat().st_size if partial.is_file() else 0
        remaining += max(0, (job.size or job.estimate) - have)
    free = shutil.disk_usage(root).free
    print(f"\n[PLAN] Files: {len(jobs)} | estimated remaining: {remaining/1024**3:.2f} GiB"
          f" | free: {free/1024**3:.2f} GiB\n[MANIFEST] {manifest}", flush=True)
    if args.list:
        for job in jobs:
            print(job.path, flush=True)
        return 1 if errors else 0
    if free < remaining + 1024**3:
        raise RuntimeError("Not enough free disk space for the download plan plus 1 GiB margin")
    successes = 0
    for index, job in enumerate(jobs, 1):
        print(f"\n[{index}/{len(jobs)}]", flush=True)
        try:
            download(job, args.idle_retries, args.stall_seconds)
            successes += 1
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            errors.append({"file": job.path, "url": job.url, "error": str(error)})
            print(f"[FAILED] {error}", flush=True)
    report = root / "download_report.json"
    report.write_text(json.dumps({"completed_or_existing": successes,
                                  "planned_files": len(jobs), "errors": errors}, indent=2))
    print(f"\n[FINISHED] {successes}/{len(jobs)} files; errors: {len(errors)}"
          f"\n[REPORT] {report}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Partial downloads retained; run the same command to resume.", flush=True)
        raise SystemExit(130)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[ERROR] {error}", flush=True)
        raise SystemExit(1)

"""
DATA STORE — persistent cache that survives Streamlit redeploys
================================================================
Streamlit Community Cloud's disk is wiped on every redeploy/reboot, which
has already cost us one full re-fetch. This module gives the universe
audit (and later the scanner) a store that survives that.

Three stores, same tiny interface (get / put / exists / list / size):

  LocalStore          - plain files on disk. Used on its own when no
                        GitHub data repo is configured (temporary on
                        Streamlit Cloud, permanent on your own PC).
  GitHubReleaseStore  - files kept as GitHub *Release assets* in a
                        PRIVATE data repo. Not git commits, so nothing
                        bloats any repo history; each asset can be up to
                        2 GB. Refuses to write to a public repo, because
                        republishing raw EODHD data publicly would breach
                        the personal-use data licence.
  LayeredStore        - GitHub as the source of truth, with a local copy of
                        the small files as a fallback. Raw 5-minute bars
                        are never copied locally, to stay inside
                        Streamlit's small container.

Keys look like "folder/name.ext". For GitHub, the folder becomes the
release tag (e.g. "meta", "results") and raw-bar files are sharded by
first letter ("bars5m-C" holds CBA.parquet, CSL.parquet, ...) to stay
well under GitHub's 1,000-assets-per-release limit.
"""
import os
import time

import requests


class StoreError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Local disk
# --------------------------------------------------------------------------
class LocalStore:
    kind = "local"

    def __init__(self, root="universe_audit_data"):
        self.root = root

    def _path(self, key):
        return os.path.join(self.root, *key.split("/"))

    def exists(self, key):
        return os.path.exists(self._path(key))

    def get(self, key):
        p = self._path(key)
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return f.read()

    def put(self, key, data):
        p = self._path(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, p)  # atomic: a crash never leaves half a file

    def size(self, key):
        p = self._path(key)
        return os.path.getsize(p) if os.path.exists(p) else None

    def list(self, prefix=""):
        out = []
        if not os.path.exists(self.root):
            return out
        for dirpath, _, files in os.walk(self.root):
            for fn in files:
                if fn.endswith(".tmp"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fn), self.root).replace(os.sep, "/")
                if rel.startswith(prefix):
                    out.append(rel)
        return sorted(out)

    def refresh(self):
        pass

    def describe(self):
        return f"local disk ({self.root})"


# --------------------------------------------------------------------------
# GitHub Releases in a private repo
# --------------------------------------------------------------------------
class GitHubReleaseStore:
    kind = "github"
    API = "https://api.github.com"
    UPLOADS = "https://uploads.github.com"
    SHARDED_FOLDERS = ("bars5m",)

    def __init__(self, repo, token, session=None, allow_public=False):
        if not repo or "/" not in repo:
            raise StoreError("DATA_REPO must look like 'owner/repo-name'.")
        self.repo = repo.strip()
        self.token = token
        self.s = session or requests.Session()
        self.allow_public = allow_public
        self._releases = {}   # tag -> release json
        self._assets = {}     # tag -> {name: asset json}
        self._checked = False

    # ---- low level -------------------------------------------------------
    def _h(self, accept="application/vnd.github+json"):
        return {"Authorization": f"Bearer {self.token}", "Accept": accept,
                "X-GitHub-Api-Version": "2022-11-28"}

    def _req(self, method, url, expect=(200,), retries=4, **kw):
        last = None
        for attempt in range(retries):
            try:
                r = self.s.request(method, url, timeout=120, **kw)
            except requests.RequestException as e:
                last = f"network error: {e}"
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code in expect:
                return r
            if r.status_code in (403, 429) and ("rate limit" in r.text.lower() or r.status_code == 429):
                wait = int(r.headers.get("Retry-After", "0") or 0) or 15 * (attempt + 1)
                last = f"GitHub rate limited ({r.status_code})"
                time.sleep(min(wait, 60))
                continue
            if r.status_code >= 500:
                last = f"GitHub {r.status_code}"
                time.sleep(3 * (attempt + 1))
                continue
            return r  # a definite client error: let the caller decide
        raise StoreError(f"GitHub request failed after {retries} tries: {method} {url} ({last})")

    def check(self):
        """Confirms the token works and the repo is PRIVATE. Called once."""
        if self._checked:
            return
        r = self._req("GET", f"{self.API}/repos/{self.repo}", expect=(200,))
        if r.status_code == 404:
            raise StoreError(f"Data repo '{self.repo}' not found, or the token can't see it.")
        if r.status_code in (401, 403):
            raise StoreError(f"GitHub token rejected for '{self.repo}' (HTTP {r.status_code}).")
        if r.status_code != 200:
            raise StoreError(f"GitHub repo check failed: HTTP {r.status_code} {r.text[:200]}")
        info = r.json()
        if not info.get("private") and not self.allow_public:
            raise StoreError(f"Data repo '{self.repo}' is PUBLIC. Make it private first - raw EODHD "
                             "data must not be republished publicly.")
        perms = info.get("permissions") or {}
        if perms and not perms.get("push", False):
            raise StoreError(f"Token can read '{self.repo}' but cannot write to it "
                             "(needs Contents: Read and write).")
        self._checked = True

    def _split(self, key):
        folder, name = key.split("/", 1)
        if "/" in name:
            raise StoreError(f"Key too deep for release storage: {key}")
        tag = f"{folder}-{name[0].upper()}" if folder in self.SHARDED_FOLDERS else folder
        return tag, name

    def _release(self, tag, create):
        if tag in self._releases:
            return self._releases[tag]
        self.check()
        r = self._req("GET", f"{self.API}/repos/{self.repo}/releases/tags/{tag}",
                      expect=(200, 404), headers=self._h())
        if r.status_code == 200:
            rel = r.json()
        elif not create:
            return None
        else:
            r = self._req("POST", f"{self.API}/repos/{self.repo}/releases", expect=(201,),
                          headers=self._h(),
                          json={"tag_name": tag, "name": tag,
                                "body": "Data cache written by the ASX universe audit. Do not edit by hand.",
                                "draft": False, "prerelease": False})
            if r.status_code != 201:
                raise StoreError(f"Could not create release '{tag}': HTTP {r.status_code} {r.text[:300]} "
                                 "(the data repo needs at least one commit - tick 'Add a README' when creating it).")
            rel = r.json()
        self._releases[tag] = rel
        return rel

    def _asset_map(self, tag, create=False):
        if tag in self._assets:
            return self._assets[tag]
        rel = self._release(tag, create=create)
        if rel is None:
            return {}
        assets, page = {}, 1
        while True:
            r = self._req("GET", f"{self.API}/repos/{self.repo}/releases/{rel['id']}/assets",
                          headers=self._h(), params={"per_page": 100, "page": page})
            if r.status_code != 200:
                raise StoreError(f"Could not list assets of '{tag}': HTTP {r.status_code}")
            batch = r.json()
            for a in batch:
                assets[a["name"]] = a
            if len(batch) < 100:
                break
            page += 1
        self._assets[tag] = assets
        return assets

    # ---- public interface ------------------------------------------------
    def refresh(self):
        """Forget cached release/asset listings (another runner may have changed them)."""
        self._releases, self._assets = {}, {}

    def exists(self, key):
        tag, name = self._split(key)
        return name in self._asset_map(tag)

    def size(self, key):
        tag, name = self._split(key)
        a = self._asset_map(tag).get(name)
        return a["size"] if a else None

    def get(self, key):
        tag, name = self._split(key)
        a = self._asset_map(tag).get(name)
        if not a:
            return None
        # GitHub answers with a redirect to a signed download URL; requests
        # drops the Authorization header on the cross-host hop, as it should.
        r = self._req("GET", f"{self.API}/repos/{self.repo}/releases/assets/{a['id']}",
                      headers=self._h("application/octet-stream"), allow_redirects=True)
        if r.status_code != 200:
            raise StoreError(f"Download of {key} failed: HTTP {r.status_code}")
        return r.content

    def put(self, key, data):
        tag, name = self._split(key)
        assets = self._asset_map(tag, create=True)
        rel = self._releases[tag]
        if name in assets:  # replace = delete then upload
            d = self._req("DELETE", f"{self.API}/repos/{self.repo}/releases/assets/{assets[name]['id']}",
                          expect=(204, 404), headers=self._h())
            if d.status_code not in (204, 404):
                raise StoreError(f"Could not replace {key}: HTTP {d.status_code}")
            assets.pop(name, None)
        r = self._req("POST", f"{self.UPLOADS}/repos/{self.repo}/releases/{rel['id']}/assets",
                      expect=(201,), params={"name": name},
                      headers={**self._h(), "Content-Type": "application/octet-stream"}, data=data)
        if r.status_code != 201:
            raise StoreError(f"Upload of {key} failed: HTTP {r.status_code} {r.text[:300]}")
        assets[name] = r.json()

    def _all_tags(self):
        self.check()
        tags, page = [], 1
        while True:
            r = self._req("GET", f"{self.API}/repos/{self.repo}/releases", headers=self._h(),
                          params={"per_page": 100, "page": page})
            if r.status_code != 200:
                raise StoreError(f"Could not list releases: HTTP {r.status_code}")
            batch = r.json()
            for rel in batch:
                self._releases.setdefault(rel["tag_name"], rel)
                tags.append(rel["tag_name"])
            if len(batch) < 100:
                break
            page += 1
        return tags

    def list(self, prefix=""):
        folder = prefix.split("/", 1)[0] if prefix else ""
        out = []
        for tag in self._all_tags():
            base = tag
            if "-" in tag and tag.split("-", 1)[0] in self.SHARDED_FOLDERS:
                base = tag.split("-", 1)[0]
            if folder and base != folder:
                continue
            for name in self._asset_map(tag):
                key = f"{base}/{name}"
                if key.startswith(prefix):
                    out.append(key)
        return sorted(out)

    def describe(self):
        return f"private GitHub repo {self.repo} (release assets)"


# --------------------------------------------------------------------------
# Local cache in front of GitHub
# --------------------------------------------------------------------------
class LayeredStore:
    kind = "layered"

    def __init__(self, local, remote, no_local_prefixes=("bars5m/",)):
        self.local, self.remote = local, remote
        self.no_local = tuple(no_local_prefixes)  # bulky raw bars never copied to Streamlit's small disk

    def _cache_ok(self, key):
        return not key.startswith(self.no_local)

    def exists(self, key):
        return self.remote.exists(key)

    def get(self, key):
        # GitHub is the source of truth (another runner, e.g. GitHub Actions,
        # may have updated it). Local copy is only a fallback if GitHub is down.
        try:
            d = self.remote.get(key)
        except StoreError:
            if not self._cache_ok(key):
                raise
            return self.local.get(key)
        if d is not None and self._cache_ok(key):
            self.local.put(key, d)
        return d

    def refresh(self):
        self.remote.refresh()

    def put(self, key, data):
        self.remote.put(key, data)  # remote first: it's the one that must not be lost
        if self._cache_ok(key):
            self.local.put(key, data)

    def size(self, key):
        return self.remote.size(key)

    def list(self, prefix=""):
        return self.remote.list(prefix)

    def describe(self):
        return self.remote.describe() + " + local cache"


def make_store(data_repo=None, data_token=None, local_root="universe_audit_data"):
    """GitHub-backed store if the data repo is configured, else local disk.
    Returns (store, warning_or_None)."""
    local = LocalStore(local_root)
    if data_repo and data_token:
        remote = GitHubReleaseStore(data_repo, data_token)
        remote.check()  # fail loudly now, not halfway through a run
        return LayeredStore(local, remote), None
    return local, ("No DATA_REPO / DATA_REPO_TOKEN configured: using temporary disk only. "
                   "A Streamlit redeploy will wipe the cache. See the setup steps to make it permanent.")

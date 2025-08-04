import json
import httpx
import re
import datetime
import pathlib
import logging

TARGETS = {
    "chrome": "https://versionhistory.googleapis.com/v1/chrome/platforms/linux/channels/stable/versions",
    "firefox": "https://product-details.mozilla.org/1.0/firefox_versions.json",
    # Safari is slower-moving; scrape Wikipedia once a quarter
    "safari": "https://en.wikipedia.org/wiki/Safari_version_history",
    "edge": "https://edgeupdates.microsoft.com/api/products?view=enterprise",
}
UA_TEMPLATE = {
    "chrome": "Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/{ver}.0.0.0 Safari/537.36",
    "edge": "Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/{ver}.0.0.0 Safari/537.36 Edg/{ver}.0.0.0",
    "firefox": "Mozilla/5.0 ({os}; rv:{ver}.0) Gecko/20100101 Firefox/{ver}.0",
    "safari": "Mozilla/5.0 ({os}) AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/{ver}.0 Safari/605.1.15",
}

OS_STRINGS = {
    "windows": "Windows NT 10.0; Win64; x64",
    "mac": "Macintosh; Intel Mac OS X 10_15_7",
    "linux": "X11; Linux x86_64",
}


def latest_chrome():
    data = httpx.get(TARGETS["chrome"]).json()
    return data["versions"][0]["version"].split(".")[0]  # "123"


def latest_edge() -> str:
    """
    Return the major version number of the latest Edge *Stable* build.

    Works with both the old and the new API response formats.
    """
    url = "https://edgeupdates.microsoft.com/api/products?view=enterprise"
    data = httpx.get(url).json()

    # 1. locate the "Stable" product bucket
    stable_block = next((p for p in data if p.get("Product") == "Stable"), None)
    if not stable_block:
        raise RuntimeError("Edge Stable block not found in API response")

    # 2. Try the old location first
    version = stable_block.get("Version")
    if version:
        return version.split(".")[0]  # "123.0.0.0" → "123"

    # 3. New format: first release entry contains the latest build
    releases = stable_block.get("Releases") or []
    if not releases:
        raise RuntimeError("No Releases array in Edge Stable block")

    version = releases[0].get("ProductVersion") or releases[0].get("Version")
    if not version:
        raise RuntimeError("Version field missing in first release entry")

    return version.split(".")[0]  # "124.0.2478.80" → "124"


def latest_firefox():
    data = httpx.get(TARGETS["firefox"]).json()
    return data["LATEST_FIREFOX_VERSION"]  # "124.0.1" → "124"


def latest_safari():
    page = httpx.get(TARGETS["safari"]).text
    m = re.search(r"Safari (\d+)\.?(\d+)?", page)
    return m.group(1) if m else "16"


def build_ua(browser, version):
    return [
        UA_TEMPLATE[browser].format(os=OS_STRINGS[o], ver=version) for o in OS_STRINGS
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    versions = {
        "chrome": latest_chrome(),
        "edge": latest_edge(),
        "firefox": latest_firefox(),
        "safari": latest_safari(),
    }
    ua_pool = sum((build_ua(b, v) for b, v in versions.items()), [])
    json_path = pathlib.Path(__file__).with_name("user_agents.json")
    json_path.write_text(json.dumps(ua_pool, indent=2))
    logging.info("Wrote %d UA strings (%s)", len(ua_pool), datetime.date.today())

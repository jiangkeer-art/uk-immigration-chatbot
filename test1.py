import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path
from datetime import datetime
import logging
import test

# ---------- 配置 ----------
BASE_URL = "https://www.gov.uk"
CHECK_INTERVAL = 3600 * 6  # 6小时检查一次
OUTPUT_DIR = "./immigration_data"
STATE_FILE = "./checkpoint.json"
LOG_FILE = "./immigration_monitor.log"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

Path(OUTPUT_DIR).mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ---------- 工具函数 ----------
def get_page(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.error(f"请求失败 {url}: {e}")
        return None


def parse_date_from_string(date_str):
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    try:
        parts = date_str.strip().split()
        if len(parts) == 3:
            day = int(parts[0])
            month = months[parts[1].lower()]
            year = int(parts[2])
            return datetime(year, month, day)
    except:
        pass
    return None


def download_file(url, filename):
    try:
        filepath = os.path.join(OUTPUT_DIR, filename)
        logger.info(f"downloaded: {url}")
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        resp.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"saved: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"failed to download {filename}: {e}")
        return None


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "last_change_hc": None,
        "last_change_date": None,
        "last_change_url": None,
        "last_archive_url": None,
        "last_archive_date": None
    }


def save_state(state):
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ---------- 核心检测逻辑 ----------
def check_statement_of_changes():
    collection_url = "https://www.gov.uk/government/collections/immigration-rules-statement-of-changes"
    logger.info(f"check the changes: {collection_url}")

    html = get_page(collection_url)
    if not html:
        return None, None, None

    soup = BeautifulSoup(html, 'html.parser')
    latest_hc = None
    latest_date = None
    latest_url = None

    for link in soup.find_all('a', href=True):
        href = link['href']
        text = link.get_text(strip=True)
        if 'Statement of changes to the Immigration Rules: HC' in text:
            match = re.search(r'HC\s+(\d+),\s+([\d\sA-Za-z]+)', text)
            if match:
                hc_num = match.group(1)
                date_str = match.group(2).strip()
                date_obj = parse_date_from_string(date_str)
                if date_obj and (latest_date is None or date_obj > latest_date):
                    latest_hc = hc_num
                    latest_date = date_obj
                    if href.startswith('/'):
                        href = urljoin(BASE_URL, href)
                    latest_url = href

    if latest_hc:
        logger.info(f"latest changes: HC {latest_hc}, {latest_date.strftime('%Y-%m-%d')}")
        logger.info(f"   URL: {latest_url}")
        return latest_hc, latest_date, latest_url

    logger.warning("dont have any changes")
    return None, None, None


def download_change_html(publication_url, hc_num, date_obj):
    logger.info(f" {hc_num} HTML")
    html = get_page(publication_url)
    if not html:
        return

    soup = BeautifulSoup(html, 'html.parser')
    html_url = None

    # 提取 accessible HTML 链接
    for link in soup.find_all('a', href=True):
        href = link['href']
        if '/accessible' in href or href.endswith('-accessible'):
            if not href.startswith('http'):
                href = urljoin(BASE_URL, href)
            html_url = href
            break

    if html_url:
        html_content = get_page(html_url)
        if html_content:
            filename = f"statement_of_change_HC{hc_num}_{date_obj.strftime('%Y%m%d')}.html"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"HTML saved: {filepath}")
        else:
            logger.warning("can not get the html")
    else:
        logger.warning("can not get the url")


def check_archive_rules():
    archive_collection = "https://www.gov.uk/government/collections/archive-immigration-rules"
    logger.info(f"check the rules: {archive_collection}")

    html = get_page(archive_collection)
    if not html:
        return None, None

    soup = BeautifulSoup(html, 'html.parser')
    latest_url = None
    latest_date = None

    for link in soup.find_all('a', href=True):
        href = link['href']
        if '/government/publications/immigration-rules-archive-' in href:
            match = re.search(r'to-(\d{1,2}-[a-z]+-\d{4})', href, re.IGNORECASE)
            if match:
                date_str = match.group(1).replace('-', ' ')
                date_obj = parse_date_from_string(date_str)
                if date_obj and (latest_date is None or date_obj > latest_date):
                    latest_date = date_obj
                    if href.startswith('/'):
                        latest_url = urljoin(BASE_URL, href)

    if latest_url:
        logger.info(f"latest rules: {latest_date.strftime('%Y-%m-%d')}")
        logger.info(f"   URL: {latest_url}")
        return latest_url, latest_date

    logger.warning("didn't have any changes")
    return None, None


def download_archive_pdf(archive_url, date_obj):
    logger.info(f"download the rules ({date_obj.strftime('%Y-%m-%d')})...")
    html = get_page(archive_url)
    if not html:
        return

    soup = BeautifulSoup(html, 'html.parser')
    pdf_url = None

    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.endswith('.pdf') or ('/attachment_data/file/' in href and '.pdf' in href):
            if not href.startswith('http'):
                href = urljoin(BASE_URL, href)
            pdf_url = href
            break

    if pdf_url:
        filename = f"immigration_rules_full_{date_obj.strftime('%Y%m%d')}.pdf"
        download_file(pdf_url, filename)
    else:
        logger.warning("didn't find the pdf")


#主循环
def check_and_update():
    logger.info("=" * 60)
    logger.info(f"check the update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    state = load_state()
    a = 0
    #检查变更声明
    hc_num, change_date, change_url = check_statement_of_changes()
    if hc_num and state.get("last_change_hc") != hc_num:
        a = 1
        logger.info(f"find new changes: HC {hc_num}")
        download_change_html(change_url, hc_num, change_date)
        state["last_change_hc"] = hc_num
        state["last_change_date"] = change_date.strftime('%Y-%m-%d') if change_date else None
        state["last_change_url"] = change_url
    elif hc_num:
        logger.info(f" dont have new changes (now: HC {hc_num})")
    else:
        logger.warning("can not get new changes")

    #检查归档规则
    archive_url, archive_date = check_archive_rules()
    if archive_url and state.get("last_archive_url") != archive_url:
        a = 1
        logger.info(f"find new rules: {archive_date.strftime('%Y-%m-%d') if archive_date else ' '}")
        download_archive_pdf(archive_url, archive_date)
        state["last_archive_url"] = archive_url
        state["last_archive_date"] = archive_date.strftime('%Y-%m-%d') if archive_date else None
    elif archive_url:
        logger.info(f"dont have new rules")
    else:
        logger.warning("can not get new rules")

    if a ==1:
        test.main()
        a = 0

    save_state(state)
    logger.info("Finished")
    logger.info("=" * 60)


def main():
    logger.info(f"saved in : {OUTPUT_DIR}")
    logger.info(f"Inspection interval: {CHECK_INTERVAL} seconds ({CHECK_INTERVAL/3600:.1f} hours)")

    check_and_update()

    while True:
        try:
            time.sleep(CHECK_INTERVAL)
            check_and_update()
        except KeyboardInterrupt:
            logger.info("stopped")
            break
        except Exception as e:
            logger.error(f"error: {e}")
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
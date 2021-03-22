import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
from lxml import etree

DATE_FMT = "%Y-%m-%d"

DateStr = Union[
    datetime, str
]  # date could be represented as a string in the chosen format
Valuation = Tuple[np.datetime64, float]

START_DATE = datetime.strftime(datetime.now() - 10 * timedelta(days=365), DATE_FMT)
VALUE_KEY = "Unit Net Value"

def get_etf_content(
    code: str, start_date: datetime, end_date: datetime, page: int = 0
) -> str:
    """Build URL to retrieve data of ETF"""
    start = datetime.strftime(start_date, DATE_FMT)
    end = datetime.strftime(end_date, DATE_FMT)

    url = f"https://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={code}&sdate={start}&edate={end}&per=25"
    if page:
        url += f"&page={page}"

    resp = requests.get(url)
    return resp.text


def translate_header(chinese: str) -> str:
    """Translate header from Chinese to English"""
    mapping = {
        "净值日期": "Date",
        "日期": "Date",
        "单位净值": VALUE_KEY,
        "累计净值": "Cumulative Net Value",
        "日增长率": "Daily Growth",
        "申购状态": "Purchase Status",
        "赎回状态": "Selling Status",
        "分红送配": "Dividends",
    }
    return mapping.get(chinese, chinese)


def convert_value(value):
    if value is None:
        return None

    if isinstance(value, str) and value.endswith("%"):
        try:
            return float(value[:-1]) / 100
        except ValueError:
            pass

    try:
        return float(value)
    except ValueError:
        pass

    try:
        return datetime.strptime(value, DATE_FMT)
    except ValueError:
        return value


def iter_rows_etf(
    code: str,
    start_date: DateStr,
    end_date: DateStr = None,
    require_start_value: bool = False,
) -> Generator[List[Any], None, None]:
    """Iter over all rows for ETF data"""

    def get_date_as_datetime(date: DateStr) -> datetime:
        return datetime.strptime(date, DATE_FMT) if isinstance(date, str) else date

    start_date = get_date_as_datetime(start_date)
    if not end_date:
        end_date = datetime.now()
    else:
        end_date = get_date_as_datetime(end_date)

    content = get_etf_content(code, start_date, end_date)
    html = etree.HTML(content)

    # We will retrieve headers (and exclude from all other search)
    headers = [translate_header(h.text) for h in html.xpath("//tr/th")]
    yield headers

    # The first date might not be included because no valuation during weekend
    # We will have to search to older values to provide it and change the date
    start = datetime.strftime(start_date, DATE_FMT)
    if require_start_value and start not in content:
        new_start_date = start_date - timedelta(days=3)
        prev_values = list(iter_rows_etf(code, new_start_date, start_date))
        if len(prev_values) > 1:
            last_values = prev_values[-1]
            yield [start] + last_values[1:]

    m_pages = re.search(r"pages:(\d+)\b", content)
    nb_pages = int(m_pages.group(1)) if m_pages else 0

    # All the data are returned from more recent to older
    # We will have to reverse both for pages and results in each page
    for page in range(nb_pages, 0, -1):
        content = get_etf_content(code, start_date, end_date, page=page)
        html = etree.HTML(content)
        # We will parse all rows except the first one
        for row in reversed(html.xpath("//tr[not(th)]")):
            yield [convert_value(col.text) for col in row.xpath("td")]


def get_etf_dataframe(
    code: str, start_date: DateStr, require_start_value: bool = False
) -> pd.DataFrame:
    """Retrieve Dataframe for an ETF"""
    rows = iter_rows_etf(code, start_date, require_start_value=require_start_value)
    headers = next(rows)
    return pd.DataFrame(rows, columns=headers)


@dataclass
class ETFData:
    """Data for ETF"""

    code: str
    path: Optional[Path] = None
    df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        if not self.path:
            self.path = Path("data") / f"{self.code}.json"

        if self.df.empty and self.path.is_file():
            self.df = pd.read_json(self.path)

    @property
    def first_date(self) -> Optional[np.datetime64]:
        """Get date of first valuation"""
        return None if self.df.empty else self.df.iloc[0]["Date"]

    @property
    def last_date(self) -> Optional[np.datetime64]:
        """Get date of latest valuation"""
        return None if self.df.empty else max(self.df["Date"])

    def update(self) -> "ETFData":
        """Update data with latest value"""
        # Just to be safe, we will remove the data of latest date
        if self.df.empty:
            self.df = get_etf_dataframe(self.code, START_DATE, True)

        else:
            last_date = self.last_date or START_DATE
            rows_last_date = self.df[self.df["Date"] == last_date]
            self.df.drop(rows_last_date.index, inplace=True)

            new_data = get_etf_dataframe(self.code, last_date, True)
            self.df = pd.concat([self.df, new_data], ignore_index=True)

        self.save()
        return self

    def save(self) -> None:
        self.df.to_json(self.path)

    def get_relative_df(self, ref_date: datetime) -> pd.DataFrame:
        """Return DataFrame relative to start value and a column with ETF code"""
        df = self.df[self.df["Date"] >= ref_date]
        if not df.empty:
            ref_value = float(df.iloc[0][VALUE_KEY])
            df["Relative Net Value"] = df[VALUE_KEY] / ref_value - 1
        df["ETF"] = self.code
        return df


ETFs = Iterable[ETFData]
ETFbyCode = Dict[str, ETFData]


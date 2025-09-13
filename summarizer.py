import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
import PyPDF2
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from jinja2 import Template
from weasyprint import HTML
import pandas as pd
import matplotlib
matplotlib.use("Agg") #backend noninteractive yapmak için yoksa terminale uyarı atıyor 
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import math


load_dotenv()



# Utility functions
# ------------------------------
def load_file(path: str) -> str:
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "".join(page.extract_text() or "" for page in reader.pages)

def safe_num(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            return None
        return float(v)
    try:
        s = str(v).replace(",", "").replace("$", "").replace("(", "-").replace(")", "")
        return float(s)
    except Exception:
        return None

def ensure_int_year(y: Any) -> Optional[int]:
    try:
        return int(y)
    except Exception:
        return None

def format_usd(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return "${:,.2f}".format(v)


# Chart generation
# ------------------------------
def generate_financial_charts(historical_financials: List[dict], company: str):
    df = pd.DataFrame(historical_financials)
    if "Year" in df.columns:
        df["Year"] = df["Year"].apply(ensure_int_year)
    for col in ["Total Revenue", "Net Income", "Total Assets", "Total Liabilities", "Cash Flow", "Equity"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_num)
        else:
            df[col] = None
    df = df.sort_values("Year").reset_index(drop=True)

    chart_paths = {}
    desktop_path = Path.home() / "Desktop"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Revenue & Net Income chart
    try:
        fig, ax = plt.subplots(figsize=(10,6))
        plot_df = df.set_index("Year")[["Total Revenue", "Net Income"]].dropna(how='all')
        if not plot_df.empty:
            # Max Value Check
            max_val = plot_df[['Total Revenue', 'Net Income']].max().max()

            # Data Scale
            if max_val >= 1_000_000_000:
                plot_df = plot_df / 1_000_000_000
                y_label = "USD (billions)"
            elif max_val >= 1_000_000:
                plot_df = plot_df / 1_000_000
                y_label = "USD (millions)"
            else:
                y_df = plot_df
                y_label = "USD"
            
            plot_df.plot(kind="bar", ax=ax, color=["#1f77b4", "#2ca02c"])
            ax.set_title("Revenue & Net Income Over Years", fontsize=14, fontweight="bold")
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_xlabel("Year", fontsize=12)
            ax.legend(loc="best", fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.xticks(rotation=0, fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            revenue_chart = desktop_path / f"{company}_revenue_netincome_{timestamp}.png"
            fig.savefig(revenue_chart, dpi=150)
            plt.close(fig)
            chart_paths["revenue_netincome"] = str(revenue_chart)
    except Exception as e:
        print("Chart error:", e)

    # YoY changes chart
    try:
        yoy_df = df.set_index("Year")[["Total Revenue", "Net Income"]]
        yoy = yoy_df.pct_change() * 100
        if not yoy.dropna(how='all').empty:
            fig2, ax2 = plt.subplots(figsize=(10,5))
            yoy.plot(kind="line", marker="o", ax=ax2, color=["#ff7f0e", "#d62728"], linewidth=2)
            ax2.set_title("Year-over-Year % Change", fontsize=14, fontweight="bold")
            ax2.set_ylabel("Percent (%)", fontsize=12)
            ax2.set_xlabel("Year", fontsize=12)
            ax2.legend(loc="best", fontsize=10)
            ax2.grid(True, linestyle="--", alpha=0.7)
            plt.xticks(rotation=0, fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            yoy_chart = desktop_path / f"{company}_yoy_changes_{timestamp}.png"
            fig2.savefig(yoy_chart, dpi=150)
            plt.close(fig2)
            chart_paths["yoy_changes"] = str(yoy_chart)
    except Exception as e:
        print("YoY chart error:", e)

    return chart_paths, df


def save_pdf(html_content: str, filename: str):
    desktop_path = Path.home() / "Desktop" / filename
    try:
        HTML(string=html_content).write_pdf(desktop_path)
        print(f"Saved PDF to Desktop: {desktop_path}")
    except Exception as e:
        print(f"Could not save PDF to Desktop: {e}")


# GenAI client api bağlantısı
# -----------------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# Model kemik görünüm
# ---------------------------
class AnnualReport(BaseModel):
    company_name: Optional[str] = None
    cik: Optional[str] = None
    fiscal_year_end: Optional[datetime] = None
    filing_date: Optional[datetime] = None
    total_revenue: Optional[float] = None
    net_income: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    num_employees: Optional[int] = None
    auditor: Optional[str] = None
    business_description: Optional[str] = None
    risk_factors: Optional[List[str]] = None
    management_discussion: Optional[str] = None
    executive_summary: Optional[str] = None
    segment_performance: Optional[List[Dict[str, Any]]] = None
    insights: Optional[List[str]] = None
    opportunities: Optional[List[str]] = None
    risks: Optional[List[str]] = None
    takeaways: Optional[List[str]] = None
    historical_financials: Optional[List[Dict[str, Any]]] = None

class EightKReport(BaseModel):
    company_name: Optional[str] = None
    cik: Optional[str] = None
    filing_date: Optional[datetime] = None
    event_description: Optional[str] = None
    impact: Optional[str] = None
    insights: Optional[List[str]] = None
    opportunities: Optional[List[str]] = None
    risks: Optional[List[str]] = None
    takeaways: Optional[List[str]] = None



# Summarizer
# -------------------------
def summarize_10k_report(file_path: str) -> AnnualReport:
    text = load_file(file_path)

    prompt = f"""
You are a financial analyst. Analyze the following annual report (10-K) and produce structured output in JSON format.
The document contains the full text of a 10-K report.

Here are the specific instructions for extracting information:
- **Financial Highlights:** Look for the latest year's values in the "Consolidated Statements of Operations" and "Consolidated Balance Sheets".
- **Historical Financials:** Find the historical financial table, typically in "Part II, Item 8. Financial Statements and Supplementary Data". Extract the data row-by-row for at least the last 3 years. The table must include a "Year" column.
- **Insights:** Base your insights on a holistic analysis of the entire document.
- **Opportunities & Risks:** Extract opportunities and risks directly from sections like "Management’s Discussion and Analysis of Financial Condition and Results of Operations" and "Risk Factors".
- **Takeaways:** Provide a high-level summary of the most important points.

Here's the information to extract with specific formatting rules:
- company_name: Extract the exact company name.
- cik: Extract the CIK number.
- fiscal_year_end: Extract the fiscal year end date in YYYY-MM-DD format.
- filing_date: Extract the filing date in YYYY-MM-DD format.
- total_revenue: Latest year's total revenue as a numeric value.
- net_income: Latest year's net income as a numeric value.
- total_assets: Latest year's total assets as a numeric value.
- total_liabilities: Latest year's total liabilities as a numeric value.
- operating_cash_flow: Latest year's operating cash flow as a numeric value.
- cash_and_equivalents: Latest year's cash and equivalents as a numeric value.
- executive_summary: Provide a comprehensive summary of the key findings, including the business overview and financial performance.
- insights: A list of 3 key insights.
- opportunities: A list of 2 key opportunities.
- risks: A list of 2 key risks.
- takeaways: A list of 3 key takeaways.
- historical_financials: A list of dictionaries, with each dictionary representing a year's data. Each dictionary should have the keys: "Year", "Total Revenue", "Net Income", "Total Assets", "Total Liabilities", "Equity", and "Cash Flow". The values must be numeric.

Report text:
{text}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            }
        )
        data = json.loads(response.text)
        
        # Eğer veri bir listeyse ilk elemanı al değilse direk kullan
        if isinstance(data, list) and len(data) > 0:
            ar = AnnualReport.model_validate(data[0])
        elif isinstance(data, dict):
            ar = AnnualReport.model_validate(data)
        else:
            raise ValidationError("Unexpected JSON format from AI response")
    except (ValidationError, Exception) as e:
        print(f"Error processing AI response: {e}")
        # Hata durumunda varsayılan bir AnnualReport nesnesi döndür
        ar = AnnualReport()

    # Chart ha
    chart_path, yoy_path = None, None
    hist = ar.historical_financials or []
    if hist:
        company_name_for_chart = ar.company_name.replace(" ", "_") if ar.company_name else "unknown_company"
        chart_paths, _ = generate_financial_charts(hist, company_name_for_chart)
        chart_path = chart_paths.get("revenue_netincome")
        yoy_path = chart_paths.get("yoy_changes")

    # Dashboard-style template
    template_str = """
    <html>
    <head>
    <meta charset="utf-8" />
    <style>
        body { font-family: Arial, sans-serif; color:#222; padding:28px; }
        h1 { color:#0b3d91; }
        h2 { margin-top:24px; color:#003366; }
        .section { margin-top:20px; }
        .summary {
            background:#eef6ff;
            border-left:6px solid #0b3d91;
            padding:16px;
            border-radius:6px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }
        .card {
            background:#f9f9f9;
            border-radius:8px;
            padding:16px;
            text-align:center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h3 { margin-bottom:10px; font-size:1.1em; }
        .card p { font-size:1.2em; font-weight:bold; }
        .card.positive { border-top:4px solid #2e8b57; }
        .card.neutral { border-top:4px solid #4682b4; }
        img { max-width:100%; margin-top:12px; }
    </style>
    </head>
    <body>
        <h1>{{ company_name }} Annual Report ({{ year }})</h1>
        <p><b>CIK:</b> {{ cik }} | <b>Filing Date:</b> {{ filing_date }}</p>

        <div class="section">
            <h2>Executive Summary</h2>
            <div class="summary">
                <p>{{ executive_summary }}</p>
            </div>
        </div>

        <div class="section">
            <h2>Financial Highlights</h2>
            <div class="grid">
                <div class="card positive"><h3>Total Revenue</h3><p>{{ total_revenue_display }}</p></div>
                <div class="card positive"><h3>Net Income</h3><p>{{ net_income_display }}</p></div>
                <div class="card neutral"><h3>Total Assets</h3><p>{{ total_assets_display }}</p></div>
                <div class="card neutral"><h3>Total Liabilities</h3><p>{{ total_liabilities_display }}</p></div>
                <div class="card neutral"><h3>Operating Cash Flow</h3><p>{{ operating_cash_flow_display }}</p></div>
                <div class="card neutral"><h3>Cash & Equivalents</h3><p>{{ cash_and_equivalents_display }}</p></div>
            </div>
        </div>

        {% if historical_financials %}
        <div class="section">
            <h2>Historical Financials</h2>
            <table border="1" cellspacing="0" cellpadding="6">
                <tr><th>Year</th><th>Total Revenue</th><th>Net Income</th><th>Assets</th><th>Liabilities</th><th>Equity</th><th>Cash Flow</th></tr>
                {% for r in historical_financials %}
                <tr>
                    <td>{{ r["Year"] }}</td>
                    <td>{{ r["Total Revenue"] }}</td>
                    <td>{{ r["Net Income"] }}</td>
                    <td>{{ r["Total Assets"] }}</td>
                    <td>{{ r["Total Liabilities"] }}</td>
                    <td>{{ r["Equity"] }}</td>
                    <td>{{ r["Cash Flow"] }}</td>
                </tr>
                {% endfor %}
            </table>
            {% if chart_path %}<img src="{{ chart_path }}" alt="Revenue & Net Income">{% endif %}
            {% if yoy_path %}<img src="{{ yoy_path }}" alt="YoY Changes">{% endif %}
        </div>
        {% endif %}

        <div class="section">
            <h2>AI Insights</h2>
            <ul>{% for i in insights %}<li>{{ i }}</li>{% endfor %}</ul>
        </div>
    </body>
    </html>
    """

    template = Template(template_str)

    def to_display(v):
        return format_usd(safe_num(v))

    html_content = template.render(
    company_name=ar.company_name,
    year=ar.fiscal_year_end.year if ar.fiscal_year_end else "N/A",
    cik=ar.cik,
    filing_date=ar.filing_date.strftime("%Y-%m-%d") if ar.filing_date else "N/A",
    executive_summary=ar.executive_summary or "",
    total_revenue_display=to_display(ar.total_revenue),
    net_income_display=to_display(ar.net_income),
    total_assets_display=to_display(ar.total_assets),
    total_liabilities_display=to_display(ar.total_liabilities),
    operating_cash_flow_display=to_display(ar.operating_cash_flow),
    cash_and_equivalents_display=to_display(ar.cash_and_equivalents),
    historical_financials=hist,
    insights=ar.insights or [],
    chart_path=chart_path,
    yoy_path=yoy_path
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    company_name_safe = ar.company_name.replace(' ', '_') if ar.company_name else "unknown_company"
    year_safe = ar.fiscal_year_end.year if ar.fiscal_year_end else "unknown_year"
    filename = f"annual_report_{company_name_safe}_{year_safe}_{timestamp}__pro.pdf"
    
    save_pdf(html_content, filename)
    return ar

def summarize_8k_report(file_path: str) -> EightKReport:
    text = load_file(file_path)
    
    # 8-K için daha detaylı ve görsel bir prompt
    prompt = f"""
You are a financial analyst. Analyze the following current report (8-K) and produce structured output in JSON format.
The document contains the full text of an 8-K report, which reports major corporate events.

Here are the specific instructions for extracting and analyzing the information:
- **Event Description:** Provide a detailed summary of the event reported, based on the relevant "Item" sections.
- **Impact:** Based on the event and its potential financial or operational consequences, provide a single-word assessment of its impact from the following options: "Very Positive", "Positive", "Neutral", "Negative", "Very Negative".
- **Insights:** Provide a list of 3 key insights derived from the event and its context within the report.
- **Opportunities:** Identify a list of 2 potential opportunities for the company resulting from this event.
- **Risks:** Identify a list of 2 potential risks or challenges associated with this event.
- **Takeaways:** Summarize the top 3 most important takeaways from the report for an investor.

Here's the information to extract with specific formatting rules:
- company_name: Extract the exact company name.
- cik: Extract the CIK number.
- filing_date: Return the filing date in YYYY-MM-DD format.
- event_description: A detailed description of the event.
- impact: A single word from the specified list (e.g., "Positive").
- insights: A list of 3 strings.
- opportunities: A list of 2 strings.
- risks: A list of 2 strings.
- takeaways: A list of 3 strings.

Report text:
{text}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            }
        )

        # aidan dönen jsona yükl
        data = json.loads(response.text)
        
        # Eğer veri bir listeyse ilk elemanı al
        if isinstance(data, list) and len(data) > 0:
            ek = EightKReport.model_validate(data[0])
        elif isinstance(data, dict):
            ek = EightKReport.model_validate(data)
        else:
            raise ValidationError("Unexpected JSON format from AI response")

    except (ValidationError, Exception) as e:
        print(f"Error processing AI response: {e}")
        ek = EightKReport()

   

    # Impacte göre kart rengi test lazım
    impact_class_map = {
        "very positive": "positive",
        "positive": "positive",
        "neutral": "neutral",
        "negative": "negative",
        "very negative": "negative"
    }
    impact_class = impact_class_map.get(ek.impact.lower() if ek.impact else "neutral", "neutral")

    # 8-K için yeni görsel şablon
    template_str = """<html>
    <head>
    <meta charset="utf-8" />
    <style>
        body { font-family: Arial, sans-serif; color:#222; padding:28px; }
        h1 { color:#8B0000; }
        h2 { margin-top:24px; color:#003366; }
        .section { margin-top:20px; }
        .summary {
            background:#f0f4f8;
            border-left:6px solid #8B0000;
            padding:16px;
            border-radius:6px;
        }
        .card {
            background:#f9f9f9;
            border-radius:8px;
            padding:16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .impact-card {
            padding:16px;
            border-radius:8px;
            color: white;
            font-weight: bold;
            text-align: center;
        }
        .impact-card.positive { background-color: #2e8b57; }
        .impact-card.neutral { background-color: #4682b4; }
        .impact-card.negative { background-color: #cc0000; }
        ul { list-style-type: none; padding-left: 0; }
        li { margin-bottom: 8px; border-left: 4px solid #ddd; padding-left: 10px; }
    </style>
    </head>
    <body>
        <h1>{{ company_name }} 8-K Report</h1>
        <p><b>CIK:</b> {{ cik }} | <b>Filing Date:</b> {{ filing_date }}</p>

        <div class="section">
            <h2>Event Description</h2>
            <div class="summary"><p>{{ event_description }}</p></div>
        </div>

        <div class="section">
            <h2>Overall Impact</h2>
            <div class="impact-card {{ impact_class }}">{{ impact }}</div>
        </div>

        <div class="section">
            <h2>AI Insights & Analysis</h2>
            <div class="grid">
                <div class="card">
                    <h3>Key Insights</h3>
                    <ul>{% for i in insights %}<li>{{ i }}</li>{% endfor %}</ul>
                </div>
                <div class="card">
                    <h3>Opportunities</h3>
                    <ul>{% for o in opportunities %}<li>{{ o }}</li>{% endfor %}</ul>
                </div>
                <div class="card">
                    <h3>Risks</h3>
                    <ul>{% for r in risks %}<li>{{ r }}</li>{% endfor %}</ul>
                </div>
                <div class="card">
                    <h3>Key Takeaways</h3>
                    <ul>{% for t in takeaways %}<li>{{ t }}</li>{% endfor %}</ul>
                </div>
            </div>
        </div>
    </body>
    </html>"""

    template = Template(template_str)

    html_content = template.render(
        company_name=ek.company_name,
        cik=ek.cik,
        filing_date=ek.filing_date.strftime("%Y-%m-%d") if ek.filing_date else "N/A",
        event_description=ek.event_description or "N/A",
        impact=ek.impact or "N/A",
        impact_class=impact_class,
        insights=ek.insights or [],
        opportunities=ek.opportunities or [],
        risks=ek.risks or [],
        takeaways=ek.takeaways or []
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    company_name_safe = ek.company_name.replace(' ', '_') if ek.company_name else "unknown_company"
    year_safe = ek.filing_date.year if ek.filing_date else "unknown_year"
    filename = f"8k_report_{company_name_safe}_{year_safe}_{timestamp}_pro.pdf"
    
    save_pdf(html_content, filename)
    return ek


# Example usage ama guiyle deniyom
# ------------------------------
if __name__ == "__main__":
    # summarize_8k_report("sample_8k.pdf")
    # summarize_10k_report("meta_10k.pdf")
    pass
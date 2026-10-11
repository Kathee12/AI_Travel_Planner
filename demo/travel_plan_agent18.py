from typing import TypedDict, Annotated, Literal
import os
import operator
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.utilities import SerpAPIWrapper  # 暂时切换到 Tavily
from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
import streamlit as st
import re
import base64
from icalendar import Calendar, Event
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
import plotly.express as px
import requests
import pandas as pd
import urllib.parse
import folium
from folium.plugins import AntPath
from streamlit_folium import st_folium
from openai import OpenAI as StepOpenAI
from pathlib import Path
from dotenv import load_dotenv

# 从项目根目录的 .env 文件加载环境变量
load_dotenv(Path(__file__).parent / ".env")


# ============================================================
#  模型常量 (gemini-2.0-flash-lite 已下线, 统一用新版)
# ============================================================
HELPER_MODEL = "gemini-2.5-flash-lite"   # 辅助任务: 地图、打包、聊天、照片识别
MAIN_MODEL   = "gemini-2.5-flash"        # 主任务: 研究、规划、修改


# ==================== State Definition ====================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    destination: str
    num_days: int
    budget: float
    final_itinerary: str
    user_feedback: str
    feedback_analysis: dict
    iteration: int
    packing_list: str
    weather_data: dict
    user_preferences: dict
    restaurant_recommendations: list
    city_guides: dict
    itinerary_json: dict


# ==================== Budget Calculator ====================
def estimate_budget(destination: str, num_days: int, budget_level: str = "medium") -> dict:
    destination_base_costs = {
        "东京": {"low": 80, "medium": 150, "high": 300},
        "tokyo": {"low": 80, "medium": 150, "high": 300},
        "巴黎": {"low": 90, "medium": 180, "high": 350},
        "paris": {"low": 90, "medium": 180, "high": 350},
        "纽约": {"low": 100, "medium": 200, "high": 400},
        "new york": {"low": 100, "medium": 200, "high": 400},
        "曼谷": {"low": 40, "medium": 80, "high": 150},
        "bangkok": {"low": 40, "medium": 80, "high": 150},
        "伦敦": {"low": 95, "medium": 190, "high": 380},
        "london": {"low": 95, "medium": 190, "high": 380},
        "default": {"low": 60, "medium": 120, "high": 250}
    }
    dest_lower = destination.lower()
    daily_cost = destination_base_costs.get(dest_lower, destination_base_costs["default"])[budget_level]
    total_budget = daily_cost * num_days
    breakdown = {
        "accommodation": total_budget * 0.35,
        "food": total_budget * 0.25,
        "transportation": total_budget * 0.15,
        "activities": total_budget * 0.20,
        "miscellaneous": total_budget * 0.05
    }
    return {"total": total_budget, "daily": daily_cost, "breakdown": breakdown}


def format_budget_display(budget_info: dict, currency: str = "USD") -> str:
    symbols = {"USD": "$", "CNY": "¥", "EUR": "€", "GBP": "£", "HKD": "HK$", "JPY": "¥", "AUD": "A$", "CAD": "C$"}
    symbol = symbols.get(currency, "$")
    breakdown = budget_info["breakdown"]
    budget_text = f"""
### Budget Planning         

**Total Budget:** {symbol}{budget_info['total']:.2f}  
**Daily Budget:** {symbol}{budget_info['daily']:.2f}

#### Budget Breakdown:
- **Accommodation:** {symbol}{breakdown['accommodation']:.2f} (35%)    
- **Food & Dining:** {symbol}{breakdown['food']:.2f} (25%)               
- **Transportation:** {symbol}{breakdown['transportation']:.2f} (15%)    
- **Activities:** {symbol}{breakdown['activities']:.2f} (20%)            
- **Miscellaneous:** {symbol}{breakdown['miscellaneous']:.2f} (5%)       

---
"""
    return budget_text


# ==================== Budget Visualization ====================
def create_budget_pie_chart(budget_info: dict):
    breakdown = budget_info['breakdown']
    labels = ['🏨 住宿', '🍽️ 吃喝', '🚗 交通', '🎭 玩乐', '🛍️ 杂项']
    values = [breakdown['accommodation'], breakdown['food'], breakdown['transportation'],
              breakdown['activities'], breakdown['miscellaneous']]
    colors = ['#FF6B8A', '#A78BFA', '#34D399', '#FBBF24', '#60A5FA']
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4, marker_colors=colors,
                                 textinfo='label+percent', textfont_size=12)])
    fig.update_layout(title_text=f"Budget Breakdown (${budget_info['total']:.0f} Total)",
                      title_font_size=16, height=400, showlegend=True)
    return fig


# ==================== Itinerary Gantt Chart ====================
def create_itinerary_gantt(itinerary: str, start_date: datetime, num_days: int):
    slot_config = {
        "Morning":   {"x0": 0,  "x1": 4,  "color": "#FBBF24", "label": "Morning"},
        "Afternoon": {"x0": 4,  "x1": 9,  "color": "#34D399", "label": "Afternoon"},
        "Evening":   {"x0": 9,  "x1": 13, "color": "#A78BFA", "label": "Evening"},
    }

    def detect_slot(text: str) -> str:
        t = text.lower()
        morning_kw  = ["morning", "breakfast", "早上", "早餐", "早晨", "am)", "am:", "8:0", "9:0", "10:0", "7:0"]
        afternoon_kw = ["afternoon", "lunch", "noon", "下午", "午餐", "12:", "13:", "14:", "1:0", "2:0", "3:0"]
        evening_kw  = ["evening", "dinner", "night", "晚上", "晚餐", "夜", "pm)", "pm:", "18:", "19:", "20:", "7:0", "8:0"]
        if any(k in t for k in afternoon_kw):
            return "Afternoon"
        if any(k in t for k in evening_kw):
            return "Evening"
        if any(k in t for k in morning_kw):
            return "Morning"
        return None

    def clean_text(s: str) -> str:
        s = re.sub(r'\*+', '', s)
        s = re.sub(r'_+', '', s)
        s = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', s)
        s = s.strip(" -:：")
        return s[:55]

    # 预处理：去掉第一个 Day/第 之前的开头废话（如 "Here is your itinerary..."）
    first_day = re.search(r'(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*{1,2})?(?:Day\s*\d+|第\s*\d+\s*天)', itinerary, re.IGNORECASE)
    if first_day:
        itinerary = itinerary[first_day.start():]

    # 匹配 Day 1 / ## Day 1 / **Day 1:** / ### **Day 1** / 第1天 / 第一天 等格式
    day_pattern = re.compile(
        r'(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*{1,2})?(?:Day\s*(\d+)|第\s*(\d+)\s*天)[^\n]*\n(.*?)(?=\n\s*(?:#{1,3}\s*)?(?:\*{1,2})?(?:Day\s*\d+|第\s*\d+\s*天)|\Z)',
        re.DOTALL | re.IGNORECASE
    )
    bars = []

    for day_match in day_pattern.finditer(itinerary):
        day_num   = int(day_match.group(1) or day_match.group(2))
        day_content = day_match.group(3)
        day_label = f"Day {day_num} ({(start_date + timedelta(days=day_num-1)).strftime('%m/%d')})"

        slot_activities = {"Morning": [], "Afternoon": [], "Evening": []}

        section_pattern = re.compile(
            r'[-•*]\s*(Morning|Afternoon|Evening|Night|早上|下午|晚上)\s*(?:\([^)]*\))?\s*[:：]\s*(.+)',
            re.IGNORECASE
        )
        for m in section_pattern.finditer(day_content):
            slot_raw = m.group(1).capitalize()
            slot = "Evening" if slot_raw in ("Night", "晚上") else \
                   "Morning"   if slot_raw in ("Morning", "早上") else \
                   "Afternoon" if slot_raw in ("Afternoon", "下午") else slot_raw
            if slot in slot_activities:
                slot_activities[slot].append(clean_text(m.group(2)))

        if not any(slot_activities.values()):
            for line in day_content.splitlines():
                line = line.strip()
                if not line or len(line) < 6:
                    continue
                if re.match(r'^#+\s', line) or re.match(r'^\*{2}(Tips?|Note|Tip)s?\*{2}', line, re.I):
                    continue
                slot = detect_slot(line)
                if slot:
                    text = re.sub(r'^[-•*\d:.\s]+', '', line)
                    text = clean_text(text)
                    if text:
                        slot_activities[slot].append(text)

        if not any(slot_activities.values()):
            all_lines = []
            for line in day_content.splitlines():
                line = line.strip()
                if not line or len(line) < 6:
                    continue
                if re.match(r'^#+\s', line) or re.match(r'^\*{1,2}(Tips?|Notes?|Tip)\*{1,2}', line, re.I):
                    continue
                if re.match(r'^[-•*]', line) or re.match(r'^\d+[.)]\s', line):
                    text = re.sub(r'^[-•*\d:.()\s]+', '', line)
                    text = clean_text(text)
                    if text:
                        all_lines.append(text)
            if all_lines:
                n = len(all_lines)
                m_end = max(1, n // 3)
                a_end = max(2, n * 2 // 3)
                slot_activities["Morning"]   = all_lines[:m_end]
                slot_activities["Afternoon"] = all_lines[m_end:a_end]
                slot_activities["Evening"]   = all_lines[a_end:]

        for slot, acts in slot_activities.items():
            if acts:
                cfg = slot_config[slot]
                label = " │ ".join(acts[:2])
                bars.append({
                    "day_label": day_label,
                    "day_num":   day_num,
                    "slot":      slot,
                    "label":     label,
                    "x0":        cfg["x0"],
                    "x1":        cfg["x1"],
                    "color":     cfg["color"],
                })

    if not bars:
        return None

    y_labels = [
        f"Day {i+1} ({(start_date + timedelta(days=i)).strftime('%m/%d')})"
        for i in range(num_days)
    ]
    y_index = {label: i for i, label in enumerate(y_labels)}

    fig = go.Figure()
    added_slots = set()

    for bar in bars:
        slot  = bar["slot"]
        yi    = y_index.get(bar["day_label"], 0)
        width = bar["x1"] - bar["x0"]
        show_legend = slot not in added_slots
        added_slots.add(slot)

        fig.add_trace(go.Bar(
            name=slot,
            x=[width],
            y=[yi],
            base=[bar["x0"]],
            orientation='h',
            marker=dict(color=bar["color"], line=dict(color="white", width=1)),
            text=bar["label"],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=11, color="white"),
            width=0.55,
            showlegend=show_legend,
            legendgroup=slot,
            hovertemplate=(
                f"<b>{bar['day_label']}</b><br>"
                f"🕐 {slot}<br>"
                f"📌 %{{text}}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text="📅 行程时间线", font=dict(size=17, color="#1E293B"), x=0.5),
        barmode="overlay",
        xaxis=dict(
            range=[-0.3, 13.3],
            tickvals=[2, 6.5, 11],
            ticktext=["☀️ Morning", "🌤 Afternoon", "🌙 Evening"],
            tickfont=dict(size=13),
            showgrid=False,
            zeroline=False,
            title="",
        ),
        yaxis=dict(
            tickvals=list(range(num_days)),
            ticktext=y_labels,
            autorange="reversed",
            tickfont=dict(size=12),
            title="",
            gridcolor="#eeeeee",
        ),
        legend=dict(
            title="",
            orientation="h",
            y=1.08,
            x=0.5,
            xanchor="center",
            font=dict(size=12),
        ),
        height=max(280, num_days * 52 + 130),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(l=140, r=20, t=90, b=30),
    )
    return fig


# ==================== Travel Statistics ====================
def calculate_statistics():
    if 'history' not in st.session_state or not st.session_state.history:
        return None
    history = st.session_state.history
    stats = {
        'total_trips': len(history),
        'total_days': sum(item['num_days'] for item in history),
        'total_budget': sum(item['budget']['total'] for item in history),
        'avg_budget': sum(item['budget']['total'] for item in history) / len(history),
        'destinations': [item['destination'] for item in history],
        'budget_breakdown': {
            'accommodation': sum(item['budget']['breakdown']['accommodation'] for item in history),
            'food': sum(item['budget']['breakdown']['food'] for item in history),
            'transportation': sum(item['budget']['breakdown']['transportation'] for item in history),
            'activities': sum(item['budget']['breakdown']['activities'] for item in history),
            'miscellaneous': sum(item['budget']['breakdown']['miscellaneous'] for item in history),
        }
    }
    return stats


def display_statistics_dashboard():
    stats = calculate_statistics()
    if not stats:
        st.info("📊 暂无统计数据，生成第一个行程后查看")
        return
    st.markdown("### 📊 旅行足迹")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trips", stats['total_trips'])
    with col2:
        st.metric("Total Days", stats['total_days'])
    with col3:
        st.metric("Total Spent", f"${stats['total_budget']:.0f}")
    with col4:
        st.metric("Avg per Trip", f"${stats['avg_budget']:.0f}")
    from collections import Counter
    dest_counter = Counter(stats['destinations'])
    most_common = dest_counter.most_common(3)
    if most_common:
        st.markdown("#### 🌍 Top Destinations")
        cols = st.columns(len(most_common))
        for idx, (dest, count) in enumerate(most_common):
            with cols[idx]:
                st.metric(dest, f"{count} trip(s)")


# ==================== Quick Templates ====================
QUICK_TEMPLATES = {
    "🌸 闺蜜短飞": {
        "icon": "🌸", "description": "3-4天 · 短途出境 · 说走就走",
        "num_days": 3, "budget_level": "low",
        "suggestions": ["Bangkok", "Singapore", "Osaka"],
        "tags": ["Citywalk", "美食", "出片"],
    },
    "🎓 毕业旅行": {
        "icon": "🎓", "description": "7-10天 · 人均5k-8k · 青春纪念",
        "num_days": 7, "budget_level": "medium",
        "suggestions": ["Tokyo", "London", "Istanbul"],
        "tags": ["打卡地标", "毕业照", "深度游"],
    },
    "💴 穷游东南亚": {
        "icon": "💴", "description": "7-10天 · 极致省钱 · 走遍东南亚",
        "num_days": 8, "budget_level": "low",
        "suggestions": ["Bangkok", "Hanoi", "Bali"],
        "tags": ["穷游", "青旅", "异国体验"],
    },
    "📸 打卡出片": {
        "icon": "📸", "description": "4-5天 · 以拍照为主线 · 氛围感满分",
        "num_days": 5, "budget_level": "medium",
        "suggestions": ["Paris", "Santorini", "Kyoto"],
        "tags": ["欧洲街拍", "咖啡馆", "神社寺庙"],
    },
}


def display_quick_templates():
    st.markdown("""
    <div class="section-header">
        <div class="icon-box" style="background: var(--pink-light);">⚡</div>
        <div class="text">快速开始</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("选一个模板，秒速生成你的专属攻略")

    template_names = list(QUICK_TEMPLATES.keys())
    cols = st.columns(len(template_names))
    for idx, template_name in enumerate(template_names):
        with cols[idx]:
            template = QUICK_TEMPLATES[template_name]
            tags_html = "".join(f'<span class="tag">{t}</span>' for t in template.get("tags", []))
            st.markdown(f"""
            <div class="template-card">
                <div class="icon">{template.get('icon', '✈️')}</div>
                <div class="title">{template_name.split(' ', 1)[-1]}</div>
                <div class="desc">{template['description']}</div>
                <div class="tags">{tags_html}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"选这个", key=f"template_{idx}", use_container_width=True):
                st.session_state.template_days = template['num_days']
                st.session_state.template_budget_level = template['budget_level']
                st.session_state.template_suggestions = template['suggestions']
                st.session_state.template_applied = True
                st.rerun()


# ==================== Weather Forecast ====================
def get_weather_forecast(city: str, api_key: str = None, num_days: int = 7) -> dict:
    if not api_key:
        import random
        return {
            "city": city,
            "forecast": [
                {
                    "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "temp_max": random.randint(20, 30),
                    "temp_min": random.randint(10, 20),
                    "condition": random.choice(["Clear", "Cloudy", "Rain", "Sunny"]),
                    "icon": random.choice(["☀️", "⛅", "🌧️", "☁️"])
                }
                for i in range(num_days)
            ]
        }
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            daily_forecast = {}
            for item in data['list']:
                date = item['dt_txt'].split()[0]
                if date not in daily_forecast:
                    daily_forecast[date] = {"temps": [], "conditions": [], "date": date}
                daily_forecast[date]["temps"].append(item['main']['temp'])
                daily_forecast[date]["conditions"].append(item['weather'][0]['main'])
            forecast = []
            for date, info in list(daily_forecast.items())[:7]:
                condition = max(set(info['conditions']), key=info['conditions'].count)
                icon_map = {"Clear": "☀️", "Clouds": "☁️", "Rain": "🌧️", "Snow": "❄️", "Thunderstorm": "⛈️"}
                forecast.append({
                    "date": date,
                    "temp_max": max(info['temps']),
                    "temp_min": min(info['temps']),
                    "condition": condition,
                    "icon": icon_map.get(condition, "🌤️")
                })
            # API 只返回约 5 天，超出部分用已有数据的均值补齐
            if len(forecast) < num_days:
                avg_max = sum(d["temp_max"] for d in forecast) / len(forecast)
                avg_min = sum(d["temp_min"] for d in forecast) / len(forecast)
                last_date = datetime.strptime(forecast[-1]["date"], "%Y-%m-%d")
                for extra in range(num_days - len(forecast)):
                    ext_date = last_date + timedelta(days=extra + 1)
                    forecast.append({
                        "date": ext_date.strftime("%Y-%m-%d"),
                        "temp_max": round(avg_max + (extra % 3 - 1)),
                        "temp_min": round(avg_min + (extra % 3 - 1)),
                        "condition": "Clouds",
                        "icon": "🌤️",
                    })
            return {"city": city, "forecast": forecast[:num_days]}
        else:
            return None
    except Exception as e:
        return None


def display_weather_forecast(destination: str, weather_api_key: str = None):
    st.markdown("### 🌤️ 天气预报")
    weather_data = get_weather_forecast(destination, weather_api_key)
    if weather_data:
        st.caption(f"7-day forecast for {weather_data['city']}")
        cols = st.columns(7)
        for idx, day in enumerate(weather_data['forecast'][:7]):
            with cols[idx]:
                date_obj = datetime.strptime(day['date'], "%Y-%m-%d")
                day_name = date_obj.strftime("%a")
                st.markdown(f"""
                <div style="text-align: center; padding: 12px 8px; background: linear-gradient(180deg, #FFF5F7, #F3F0FF);
                            border-radius: 14px; border: 1px solid #F1F5F9;">
                    <div style="font-weight: 700; font-size: 0.82rem; color: #64748B;">{day_name}</div>
                    <div style="font-size: 2rem; margin: 4px 0;">{day['icon']}</div>
                    <div style="font-size: 0.9rem; font-weight: 600; color: #1E293B;">{day['temp_max']:.0f}°/{day['temp_min']:.0f}°</div>
                    <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 2px;">{day['condition']}</div>
                </div>
                """, unsafe_allow_html=True)
        conditions = [day['condition'] for day in weather_data['forecast']]
        suggestions = []
        if "Rain" in conditions or "Thunderstorm" in conditions:
            suggestions.append("🌂 Don't forget your umbrella!")
        if any(day['temp_min'] < 15 for day in weather_data['forecast']):
            suggestions.append("🧥 Pack warm clothes")
        if any(day['temp_max'] > 28 for day in weather_data['forecast']):
            suggestions.append("🕶️ Bring sunscreen and sunglasses")
        if suggestions:
            st.info("📝 Packing tips: " + " | ".join(suggestions))
    else:
        st.info("🌍 Weather forecast unavailable. Add OpenWeatherMap API key for real-time weather data.")
    return weather_data


def render_weather_bar(destination: str, start_date: datetime, num_days: int,
                       weather_api_key: str = None):
    """在 Step 1 输入行下方渲染一条紧凑的天气概览条（纯 HTML，~60px 高）。"""
    if not destination:
        return
    weather_data = get_weather_forecast(destination, weather_api_key, num_days=num_days)
    if not weather_data or not weather_data.get("forecast"):
        return

    forecasts = weather_data["forecast"][:num_days]
    if not forecasts:
        return

    # 根据天数动态调整单元格尺寸
    n = len(forecasts)
    if n <= 7:
        cell_w, icon_sz, font_sz = "64px", "1.3rem", "0.7rem"
    elif n <= 10:
        cell_w, icon_sz, font_sz = "52px", "1.1rem", "0.65rem"
    else:
        cell_w, icon_sz, font_sz = "42px", "1rem", "0.6rem"

    # 生成天气单元格 HTML — 全部拼成单行字符串，避免 Streamlit 把换行当分块
    cells_parts = []
    for i, day in enumerate(forecasts):
        date_obj = start_date + timedelta(days=i)
        day_label = date_obj.strftime("%m/%d")
        weekday = date_obj.strftime("%a")
        icon = day.get('icon', '🌤️')
        temp = f"{day['temp_max']:.0f}°/{day['temp_min']:.0f}°"
        cells_parts.append(
            f'<div style="text-align:center;flex:1;min-width:0;">'
            f'<div style="font-size:{font_sz};color:#94A3B8;">{weekday}</div>'
            f'<div style="font-size:{font_sz};color:#64748B;font-weight:600;">{day_label}</div>'
            f'<div style="font-size:{icon_sz};line-height:1.3;">{icon}</div>'
            f'<div style="font-size:{font_sz};color:#1E293B;font-weight:600;">{temp}</div>'
            f'</div>'
        )
    cells_html = "".join(cells_parts)

    # 打包提示
    conditions = [d.get("condition", "") for d in forecasts]
    tips = []
    if any(c in ("Rain", "Thunderstorm") for c in conditions):
        tips.append("🌂 带伞")
    if any(d.get("temp_max", 25) > 28 for d in forecasts):
        tips.append("🧴 防晒")
    if any(d.get("temp_min", 20) < 15 for d in forecasts):
        tips.append("🧥 带外套")
    tips_html = f'<span style="font-size:0.72rem;color:#94A3B8;margin-left:12px;">💡 {" · ".join(tips)}</span>' if tips else ""

    bar_html = (
        '<div style="background:linear-gradient(135deg,#F8FAFC,#F1F5F9);border:1px solid #E2E8F0;'
        'border-radius:10px;padding:8px 12px;margin-top:4px;overflow:hidden;">'
        '<div style="display:flex;align-items:flex-start;gap:2px;width:100%;">'
        f'{cells_html}'
        '</div>'
        f'{tips_html}'
        '</div>'
    )
    st.markdown(bar_html, unsafe_allow_html=True)

    # 存到 session 供后续使用
    st.session_state.weather_data = weather_data


# ==================== Itinerary Comparison ====================
def display_itinerary_comparison():
    if 'history' not in st.session_state or len(st.session_state.history) < 2:
        st.info("📊 You need at least 2 trips in history to compare")
        return
    st.markdown("### 🔄 Compare Itineraries")
    history_options = {
        f"{item['destination']} - {item['num_days']}d (${item['budget']['total']:.0f})": item['id']
        for item in st.session_state.history
    }
    selected_trips = st.multiselect("Select trips to compare (2-3 recommended)",
                                    options=list(history_options.keys()), max_selections=3)
    if len(selected_trips) >= 2:
        selected_items = [item for item in st.session_state.history
                          if item['id'] in [history_options[name] for name in selected_trips]]
        comparison_data = {
            "Destination": [item['destination'] for item in selected_items],
            "Days": [item['num_days'] for item in selected_items],
            "Total Budget": [f"${item['budget']['total']:.0f}" for item in selected_items],
            "Daily Budget": [f"${item['budget']['daily']:.0f}" for item in selected_items],
            "Hotel/Day": [f"${item['budget']['breakdown']['accommodation'] / item['num_days']:.0f}"
                          for item in selected_items],
            "Food/Day": [f"${item['budget']['breakdown']['food'] / item['num_days']:.0f}"
                         for item in selected_items],
        }
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("#### 💰 Budget Comparison")
        fig = go.Figure()
        categories = ['Accommodation', 'Food', 'Transportation', 'Activities', 'Misc']
        for item in selected_items:
            breakdown = item['budget']['breakdown']
            values = [breakdown['accommodation'], breakdown['food'], breakdown['transportation'],
                      breakdown['activities'], breakdown['miscellaneous']]
            fig.add_trace(go.Bar(name=item['destination'], x=categories, y=values,
                                 text=[f"${v:.0f}" for v in values], textposition='auto'))
        fig.update_layout(barmode='group', height=400, xaxis_title="Category",
                          yaxis_title="Amount ($)", legend_title="Destination")
        st.plotly_chart(fig, use_container_width=True, key="comparison_bar_chart")
        st.markdown("#### 📈 Value Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            best_value = min(selected_items, key=lambda x: x['budget']['daily'])
            st.metric("Best Daily Value", best_value['destination'],
                      f"${best_value['budget']['daily']:.0f}/day")
        with col2:
            longest = max(selected_items, key=lambda x: x['num_days'])
            st.metric("Longest Trip", longest['destination'], f"{longest['num_days']} days")
        with col3:
            total_spent = sum(item['budget']['total'] for item in selected_items)
            st.metric("Total if Combined", f"${total_spent:.0f}",
                      f"{sum(item['num_days'] for item in selected_items)} days")
    else:
        st.caption("👆 Select at least 2 trips to see comparison")


# ==================== User Preferences Collection ====================
def collect_user_preferences():
    """平铺展示偏好选项（用于 Step 2），不再包裹 expander。"""
    st.caption("选填 · 填得越多攻略越对你口味 · 跳过也行")
    col1, col2 = st.columns(2)
    with col1:
        travel_pace = st.selectbox(
            "旅行节奏 🏃‍♀️",
            options=["Not specified", "Relaxed (2-3 activities/day)", "Moderate (4-5 activities/day)",
                     "Packed (6+ activities/day)"],
            format_func=lambda x: {"Not specified": "还没想好", "Relaxed (2-3 activities/day)": "🧘 佛系躺平（2-3个活动/天）",
                "Moderate (4-5 activities/day)": "🚶‍♀️ 正常节奏（4-5个/天）",
                "Packed (6+ activities/day)": "🏃‍♀️ 暴走打卡（6+/天）"}.get(x, x),
            help="每天想安排多少活动？"
        )
        traveling_with = st.multiselect(
            "和谁一起 👯",
            options=["Solo", "Partner/Couple", "Family with young kids", "Family with teenagers", "Friends",
                     "Elderly"],
            format_func=lambda x: {"Solo": "🙋‍♀️ 独自旅行", "Partner/Couple": "💑 和对象",
                "Family with young kids": "👶 带小孩", "Family with teenagers": "👨‍👩‍👧 家庭出游",
                "Friends": "👯 闺蜜/朋友", "Elderly": "👴 带长辈"}.get(x, x),
            help="同行小伙伴是？"
        )
        interests = st.multiselect(
            "兴趣偏好 💖",
            options=["History & Culture", "Nature & Outdoors", "Food & Dining", "Shopping",
                     "Nightlife & Entertainment", "Museums & Art", "Photography", "Adventure Sports",
                     "Beach & Relaxation", "Local Experiences"],
            format_func=lambda x: {"History & Culture": "🏛️ 历史文化", "Nature & Outdoors": "🌿 自然风光",
                "Food & Dining": "🍜 美食探店", "Shopping": "🛍️ 买买买",
                "Nightlife & Entertainment": "🌙 夜生活", "Museums & Art": "🎨 看展",
                "Photography": "📸 拍照出片", "Adventure Sports": "🏄‍♀️ 冒险运动",
                "Beach & Relaxation": "🏖️ 海滩躺平", "Local Experiences": "🏘️ 本地体验"}.get(x, x),
            help="你最想怎么玩？"
        )
    with col2:
        dietary_restrictions = st.text_input(
            "饮食要求 🥗",
            placeholder="如：素食、清真、无麸质、不吃辣...",
            help="有忌口吗？"
        )
        mobility_considerations = st.selectbox(
            "体力情况 🦵",
            options=["Not specified", "No limitations", "Prefer less walking", "Wheelchair accessible needed",
                     "Elderly-friendly"],
            format_func=lambda x: {"Not specified": "不限", "No limitations": "💪 体力无限",
                "Prefer less walking": "🚶 不想走太多路", "Wheelchair accessible needed": "♿ 需要无障碍",
                "Elderly-friendly": "👴 适合长辈"}.get(x, x),
            help="走路多了会累吗？"
        )
        wake_up_time = st.selectbox(
            "起床时间 ⏰",
            options=["Not specified", "Early bird (6-7 AM)", "Normal (8-9 AM)", "Late riser (10 AM+)"],
            format_func=lambda x: {"Not specified": "看心情", "Early bird (6-7 AM)": "🌅 早起鸟（6-7点）",
                "Normal (8-9 AM)": "☀️ 正常（8-9点）", "Late riser (10 AM+)": "😴 睡到自然醒（10点+）"}.get(x, x),
            help="旅行的时候几点起？"
        )
    additional_notes = st.text_input(
        "其他想说的 💬",
        placeholder="如：想去小众景点、爱逛咖啡馆、想体验local生活..."
    )
    preferences = {
        "travel_pace": travel_pace if travel_pace != "Not specified" else None,
        "traveling_with": traveling_with if traveling_with else None,
        "interests": interests if interests else None,
        "dietary_restrictions": dietary_restrictions if dietary_restrictions else None,
        "mobility": mobility_considerations if mobility_considerations != "Not specified" else None,
        "wake_up_time": wake_up_time if wake_up_time != "Not specified" else None,
        "additional_notes": additional_notes if additional_notes else None
    }
    return {k: v for k, v in preferences.items() if v}


def format_preferences_for_llm(preferences: dict) -> str:
    if not preferences:
        return ""
    pref_text = "\n\nUSER PREFERENCES:\n"
    if preferences.get("travel_pace"):
        pref_text += f"- Travel Pace: {preferences['travel_pace']}\n"
    if preferences.get("traveling_with"):
        pref_text += f"- Traveling With: {', '.join(preferences['traveling_with'])}\n"
    if preferences.get("interests"):
        pref_text += f"- Interests: {', '.join(preferences['interests'])}\n"
    if preferences.get("dietary_restrictions"):
        pref_text += f"- Dietary Restrictions: {preferences['dietary_restrictions']}\n"
    if preferences.get("mobility"):
        pref_text += f"- Mobility: {preferences['mobility']}\n"
    if preferences.get("wake_up_time"):
        pref_text += f"- Preferred Wake-up Time: {preferences['wake_up_time']}\n"
    if preferences.get("additional_notes"):
        pref_text += f"- Additional Notes: {preferences['additional_notes']}\n"
    return pref_text


# ==================== Packing List Generator ====================
def extract_activities_from_itinerary(itinerary: str, llm) -> list[str]:
    prompt = f"""
    Extract activity types from this itinerary. Return ONLY a JSON array of activity keywords.

    ITINERARY:
    {itinerary[:1000]}

    Example output: ["hiking", "museum visit", "fine dining", "beach swimming", "temple tour"]

    Return only the JSON array, no other text.
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        activities = json.loads(response.content.strip())
        return activities if isinstance(activities, list) else []
    except:
        keywords = ["hiking", "swim", "beach", "museum", "temple", "restaurant", "shopping", "nightlife", "formal",
                    "dinner"]
        found = []
        itinerary_lower = itinerary.lower()
        for keyword in keywords:
            if keyword in itinerary_lower:
                found.append(keyword)
        return found


def generate_packing_list(destination: str, num_days: int, weather_data: dict, activities: list[str]) -> str:
    packing_list = {
        "✈️ Travel Essentials": [
            "Passport/ID", "Travel insurance documents", "Flight tickets/confirmations",
            "Hotel reservations", "Phone & charger", "Power bank",
            "Medications (if needed)", "Cash & credit cards"
        ],
        "👔 Clothing": [],
        "🎒 Accessories": [],
        "🏃 Activity-Specific": []
    }
    if weather_data and weather_data.get("forecast"):
        forecasts = weather_data["forecast"]
        will_rain = any(day.get("condition") in ["Rain", "Thunderstorm"] for day in forecasts)
        min_temp = min(day.get("temp_min", 20) for day in forecasts)
        max_temp = max(day.get("temp_max", 30) for day in forecasts)
        if min_temp < 10:
            packing_list["👔 Clothing"].extend([
                "Heavy jacket/coat", "Thermal underwear", "Wool sweater", "Long pants x 2", "Warm socks"
            ])
        elif min_temp < 15:
            packing_list["👔 Clothing"].extend([
                "Light jacket", "Sweater/cardigan", "Long pants x 2", "Long-sleeve shirts"
            ])
        if max_temp > 28:
            packing_list["👔 Clothing"].extend([
                "Sunscreen SPF 50+", "Sunglasses", "Sun hat/cap", "Light breathable clothes"
            ])
            packing_list["🎒 Accessories"].append("Cooling towel")
        if will_rain:
            packing_list["🎒 Accessories"].extend([
                "Compact umbrella ☔", "Waterproof jacket", "Waterproof bag cover"
            ])
    packing_list["👔 Clothing"].extend([
        f"Underwear x {num_days + 2}", f"Socks x {num_days + 2}",
        "T-shirts/casual tops x 3-4", "Comfortable walking shoes", "Sleepwear"
    ])
    activities_lower = [act.lower() for act in activities]
    if any("hik" in act or "trek" in act or "mountain" in act for act in activities_lower):
        packing_list["🏃 Activity-Specific"].extend([
            "Hiking boots/trail shoes", "Backpack (20-30L)", "Water bottle",
            "Quick-dry clothing", "First aid kit"
        ])
    if any("swim" in act or "beach" in act or "pool" in act for act in activities_lower):
        packing_list["🏃 Activity-Specific"].extend([
            "Swimsuit", "Beach towel", "Flip-flops/sandals", "Waterproof phone case"
        ])
    if any("formal" in act or "fine dining" in act or "restaurant" in act or "theater" in act
           for act in activities_lower):
        packing_list["🏃 Activity-Specific"].extend([
            "Smart casual outfit", "Dress shoes", "Belt", "Nice shirt/blouse"
        ])
    if any("temple" in act or "mosque" in act or "church" in act for act in activities_lower):
        packing_list["🏃 Activity-Specific"].extend([
            "Modest clothing (cover shoulders/knees)", "Scarf/shawl", "Socks (for shoe removal)"
        ])
    if any("photo" in act or "camera" in act for act in activities_lower):
        packing_list["🏃 Activity-Specific"].extend([
            "Camera + extra batteries", "Memory cards", "Tripod (optional)"
        ])
    result = f"# 🧳 Packing List for {destination}\n"
    result += f"**Duration:** {num_days} days\n\n"
    for category, items in packing_list.items():
        if items:
            result += f"## {category}\n"
            unique_items = []
            seen = set()
            for item in items:
                if item.lower() not in seen:
                    unique_items.append(item)
                    seen.add(item.lower())
            for item in unique_items:
                result += f"- [ ] {item}\n"
            result += "\n"
    result += "---\n*💡 Tip: Check airline baggage restrictions before packing!*\n"
    return result


# ==================== AI CHAT ASSISTANT ====================
def initialize_chat_assistant(gemini_api_key: str):
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'chat_llm' not in st.session_state:
        st.session_state.chat_llm = ChatGoogleGenerativeAI(
            model=HELPER_MODEL,
            google_api_key=gemini_api_key,
            temperature=0.7
        )


def get_chat_context() -> str:
    context = "Current Travel Planning Context:\n"
    if st.session_state.get('current_destination'):
        context += f"- Destination: {st.session_state.current_destination}\n"
    if st.session_state.get('current_days'):
        context += f"- Duration: {st.session_state.current_days} days\n"
    if st.session_state.get('budget_info'):
        context += f"- Budget: ${st.session_state.budget_info['total']:.0f}\n"
    if st.session_state.get('start_date') and st.session_state.get('end_date'):
        context += f"- Dates: {st.session_state.start_date.strftime('%Y-%m-%d')} to {st.session_state.end_date.strftime('%Y-%m-%d')}\n"
    if st.session_state.get('itinerary'):
        context += f"\nCurrent Itinerary (first 500 chars):\n{st.session_state.itinerary[:500]}...\n"
    return context


def display_chat_assistant():
    st.markdown("### 💬 旅行小助手")
    st.caption("有问题随时问我~")
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    st.markdown("**Quick Questions:**")
    quick_questions = ["💰 怎么省钱？", "🍜 有什么好吃的？",
                       "🚇 交通怎么走？", "📸 哪里适合拍照？"]
    cols = st.columns(4)
    for idx, question in enumerate(quick_questions):
        with cols[idx]:
            if st.button(question, key=f"quick_q_{idx}", use_container_width=True):
                st.session_state.pending_question = question.split(" ", 1)[1]
    user_input = st.text_input("💬 输入问题", placeholder="想问点什么...", key="chat_assistant_input")
    # 快捷问题按钮触发
    if 'pending_question' in st.session_state:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question
        st.session_state['_last_chat_input'] = ''  # 确保触发
    if user_input and user_input != st.session_state.get('_last_chat_input', ''):
        st.session_state['_last_chat_input'] = user_input
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        context = get_chat_context()
        system_prompt = f"""You are a helpful and knowledgeable travel assistant. 
You help users plan their trips by answering questions about destinations, budgets, activities, and travel tips.

{context}

Provide concise, practical, and friendly advice. If the user's question relates to their current itinerary, 
reference it specifically. Keep responses under 200 words unless more detail is needed."""
        recent_messages = st.session_state.chat_messages[-5:]
        messages = [SystemMessage(content=system_prompt)]
        for msg in recent_messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        try:
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.chat_llm.invoke(messages)
                assistant_message = response.content
                st.session_state.chat_messages.append({"role": "assistant", "content": assistant_message})
                st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
    if st.session_state.chat_messages:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()


# ==================== History Management ====================
def save_to_history(destination: str, num_days: int, budget_info: dict, itinerary: str,
                    start_date: datetime = None, end_date: datetime = None, packing_list: str = None):
    if 'history' not in st.session_state:
        st.session_state.history = []
    history_item = {
        "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "destination": destination,
        "num_days": num_days,
        "budget": budget_info,
        "itinerary": itinerary,
        "packing_list": packing_list,
        "start_date": start_date.strftime("%Y-%m-%d") if start_date else None,
        "end_date": end_date.strftime("%Y-%m-%d") if end_date else None
    }
    st.session_state.history.insert(0, history_item)
    if len(st.session_state.history) > 5:
        st.session_state.history = st.session_state.history[:5]


def delete_history_item(item_id: str):
    if 'history' in st.session_state:
        st.session_state.history = [item for item in st.session_state.history if item['id'] != item_id]


def load_history_item(item_id: str):
    if 'history' in st.session_state:
        for item in st.session_state.history:
            if item['id'] == item_id:
                st.session_state.itinerary = item['itinerary']
                st.session_state.budget_info = item['budget']
                st.session_state.current_destination = item['destination']
                st.session_state.current_days = item['num_days']
                st.session_state.packing_list = item.get('packing_list', '')
                break


def add_dates_to_itinerary(itinerary: str, start_date: datetime, num_days: int) -> str:
    lines = itinerary.split('\n')
    result = []
    for line in lines:
        match = re.match(r'^(Day\s+(\d+))(\s*[:：]?\s*)(.*)', line, re.IGNORECASE)
        if match:
            day_num = int(match.group(2))
            current_date = start_date + timedelta(days=day_num - 1)
            date_str = current_date.strftime('%B %d, %Y')
            weekday = current_date.strftime('%A')
            new_line = f"{match.group(1)} ({weekday}, {date_str}){match.group(3)}{match.group(4)}"
            result.append(new_line)
        else:
            result.append(line)
    return '\n'.join(result)


# ==================== Export Functions ====================
def format_itinerary_for_export(destination: str, num_days: int, budget_info: dict, itinerary: str,
                                format_type: str = "txt", start_date: datetime = None,
                                end_date: datetime = None, packing_list: str = None) -> str:
    if start_date:
        itinerary = add_dates_to_itinerary(itinerary, start_date, num_days)
    date_info = ""
    if start_date and end_date:
        date_info = f"\nTravel Dates: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
    if format_type == "txt":
        content = f"""
{'=' * 60}
AI TRAVEL PLANNER - ITINERARY
{'=' * 60}

Destination: {destination}
Duration: {num_days} days{date_info}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{'=' * 60}
BUDGET PLANNING
{'=' * 60}

Total Budget: ${budget_info['total']:.2f}
Daily Budget: ${budget_info['daily']:.2f}

Budget Breakdown:
  - Accommodation: ${budget_info['breakdown']['accommodation']:.2f} (35%)
  - Food & Dining: ${budget_info['breakdown']['food']:.2f} (25%)
  - Transportation: ${budget_info['breakdown']['transportation']:.2f} (15%)
  - Activities: ${budget_info['breakdown']['activities']:.2f} (20%)
  - Miscellaneous: ${budget_info['breakdown']['miscellaneous']:.2f} (5%)

{'=' * 60}
DETAILED ITINERARY
{'=' * 60}

{itinerary}
"""
        if packing_list:
            content += f"""
{'=' * 60}
PACKING LIST
{'=' * 60}

{packing_list}
"""
        content += f"""
{'=' * 60}
Generated by AI Travel Planner
Powered by LangGraph + Google Gemini
{'=' * 60}
"""
        return content
    elif format_type == "md":
        date_md = f"\n**Travel Dates:** {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}  " if start_date and end_date else ""
        content = f"""# 🌍 Travel Itinerary: {destination}

**Duration:** {num_days} days  {date_md}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 📋 Detailed Itinerary

{itinerary}
"""
        if packing_list:
            content += f"\n---\n\n{packing_list}\n"
        content += "\n---\n\n*Generated by AI Travel Planner · Powered by LangGraph + Google Gemini*\n"
        return content


# ==================== Tools Definition ====================
# serp_wrapper = None  # SerpAPI（暂停使用）
tavily_client = None


# @tool
# def search_web(query: str) -> str:
#     """Search the web for travel information using SerpAPI."""
#     try:
#         results = serp_wrapper.run(query)
#         return f"Search results for '{query}':\n{results}"
#     except Exception as e:
#         return f"Search failed: {str(e)}"

@tool
def search_web(query: str) -> str:
    """Search the web for travel information."""
    try:
        results = tavily_client.search(query, max_results=5)
        formatted = []
        for r in results.get("results", []):
            formatted.append(f"- {r.get('content', '')[:300]}")
        return f"Search results for '{query}':\n" + "\n".join(formatted)
    except Exception as e:
        return f"Search failed: {str(e)}"


# ==================== Calendar Generation ====================
def generate_ics_content(plan_text: str, start_date: datetime = None, destination: str = "") -> bytes:
    cal = Calendar()
    cal.add('prodid', '-//AI Travel Planner//langgraph//')
    cal.add('version', '2.0')
    if start_date is None:
        start_date = datetime.today()
    day_pattern = re.compile(r'Day (\d+)[:\s]+(.*?)(?=Day \d+|$)', re.DOTALL)
    days = day_pattern.findall(plan_text)
    if not days:
        event = Event()
        event.add('summary', f"Travel to {destination}" if destination else "Travel Itinerary")
        event.add('description', plan_text)
        event.add('dtstart', start_date.date())
        event.add('dtend', start_date.date())
        event.add("dtstamp", datetime.now())
        cal.add_component(event)
    else:
        for day_num, day_content in days:
            day_num = int(day_num)
            current_date = start_date + timedelta(days=day_num - 1)
            event = Event()
            event.add('summary', f"{destination} - Day {day_num}" if destination else f"Day {day_num} Itinerary")
            event.add('description', day_content.strip())
            event.add('dtstart', current_date.date())
            event.add('dtend', current_date.date())
            event.add("dtstamp", datetime.now())
            cal.add_component(event)
    return cal.to_ical()


# ==================== RESTAURANT HELPERS ====================
def build_restaurant_links(name: str, destination: str) -> dict:
    query = urllib.parse.quote_plus(f"{name} {destination}")
    maps_url = f"https://www.google.com/maps/search/{query}"
    thefork_url = f"https://www.thefork.com/search?cityName={urllib.parse.quote_plus(destination)}&searchQuery={urllib.parse.quote_plus(name)}"
    return {"maps": maps_url, "thefork": thefork_url}


PRICE_ICONS = {"$": "💚 Budget", "$$": "💛 Mid-range", "$$$": "🔴 Upscale", "$$$$": "💜 Fine Dining"}
MEAL_ICONS  = {"Breakfast": "🌅", "Lunch": "☀️", "Dinner": "🌙", "Brunch": "🍳",
               "Snack": "🍡", "Any": "🍽️"}


def display_restaurant_cards(restaurants: list[dict]):
    if not restaurants:
        return

    st.markdown("### 🍽️ Restaurant Recommendations")
    st.caption("Curated picks per city — click to open map or book a table.")

    st.markdown("""
    <style>
    .rest-card {
        background: #ffffff;
        border: 1.5px solid #F1F5F9;
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.25s ease;
    }
    .rest-card:hover {
        border-color: #FF6B8A;
        box-shadow: 0 6px 20px rgba(255,107,138,0.12);
        transform: translateY(-2px);
    }
    .rest-title { font-size: 1.05rem; font-weight: 700; color: #1E293B; margin-bottom: 2px; }
    .rest-meta  { font-size: 0.82rem; color: #64748B; margin-bottom: 6px; }
    .rest-desc  { font-size: 0.88rem; color: #475569; line-height: 1.5; margin-bottom: 10px; }
    .rest-tip   { font-size: 0.82rem; color: #A78BFA; font-style: italic; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; margin-right: 4px; }
    .badge-cuisine { background: #F3F0FF; color: #7C3AED; }
    .badge-price   { background: #FFF5F7; color: #FF6B8A; }
    .city-header {
        font-size: 1.1rem; font-weight: 700; color: #1E293B;
        border-left: 4px solid #FF6B8A; padding-left: 12px;
        margin: 20px 0 12px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    city_groups: dict[str, list] = {}
    for r in restaurants:
        city = r.get("city") or r.get("destination", "Other")
        city_groups.setdefault(city, []).append(r)

    meal_order = ["Breakfast", "Brunch", "Lunch", "Snack", "Dinner", "Any"]

    for city, recs in city_groups.items():
        st.markdown(f'<div class="city-header">📍 {city}</div>', unsafe_allow_html=True)

        meal_groups: dict[str, list] = {}
        for r in recs:
            meal = r.get("meal_type", "Any")
            meal_groups.setdefault(meal, []).append(r)

        for meal in meal_order:
            if meal not in meal_groups:
                continue
            meal_icon = MEAL_ICONS.get(meal, "🍽️")
            st.markdown(f"**{meal_icon} {meal}**")

            meal_recs = meal_groups[meal]
            cols = st.columns(min(len(meal_recs), 3))
            for idx, r in enumerate(meal_recs):
                col = cols[idx % len(cols)]
                with col:
                    price_label   = PRICE_ICONS.get(r.get("price_range", "$$"), "💛 Mid-range")
                    links         = build_restaurant_links(r["name"], city)
                    cuisine_badge = f'<span class="badge badge-cuisine">{r.get("cuisine","")}</span>'
                    price_badge   = f'<span class="badge badge-price">{r.get("price_range","$$")} {price_label}</span>'

                    st.markdown(f"""
                    <div class="rest-card">
                        <div class="rest-title">{r["name"]}</div>
                        <div class="rest-meta">{cuisine_badge}{price_badge}</div>
                        <div class="rest-desc">{r.get("description","")}</div>
                        <div class="rest-tip">{("💡 " + r["tip"]) if r.get("tip") else ""}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        st.link_button("🗺️ Maps", links["maps"], use_container_width=True)
                    with btn_col2:
                        st.link_button("📅 TheFork", links["thefork"], use_container_width=True)


# ==================== MAP FEATURE (PER-CITY) ====================

DAY_COLORS = [
    "#FF6B8A", "#A78BFA", "#34D399", "#FBBF24", "#60A5FA",
    "#F472B6", "#818CF8", "#6EE7B7", "#FCD34D", "#93C5FD",
]


def extract_locations_from_itinerary(itinerary: str, destination: str, num_days: int, llm) -> list[dict]:
    """Let LLM extract locations with real coordinates and city tags from itinerary."""
    prompt = f"""From the travel itinerary below, extract ALL specific locations/attractions/restaurants visited.

For EACH location, provide:
- name: the place name
- lat: latitude (use your best knowledge of the actual coordinates)
- lng: longitude
- day: which day number (1, 2, 3...)
- time_slot: "Morning", "Afternoon", or "Evening"
- city: which city this location is in
- description: a very brief 5-10 word description

IMPORTANT:
- Use REAL coordinates for well-known places (not made up ones)
- For lesser-known places, estimate based on the city/area
- Include restaurants and dining spots too
- Order by day number, then by time_slot (Morning → Afternoon → Evening)
- The "city" field is critical — tag each location with its actual city

Return ONLY a JSON array. No markdown, no backticks, no extra text.

DESTINATION: {destination}
DURATION: {num_days} days

ITINERARY:
{itinerary[:12000]}
"""
    try:
        response = llm.invoke([
            SystemMessage(content="You are a geocoding assistant. Return ONLY a raw JSON array, no markdown."),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
        start, end = raw.find('['), raw.rfind(']') + 1
        if start != -1 and end > start:
            locations = json.loads(raw[start:end])
            valid = []
            for loc in locations:
                if all(k in loc for k in ("name", "lat", "lng", "day")):
                    loc.setdefault("time_slot", "Morning")
                    loc.setdefault("description", "")
                    loc.setdefault("city", destination)
                    if -90 <= loc["lat"] <= 90 and -180 <= loc["lng"] <= 180:
                        valid.append(loc)
            return valid
    except Exception as ex:
        print(f"[Map] Location extraction failed: {ex}")
    return []


def create_city_map(locations: list[dict], city_name: str, start_date: datetime = None) -> folium.Map:
    """Build a folium map for a single city with day-colored markers and route lines."""
    if not locations:
        return None

    avg_lat = sum(loc["lat"] for loc in locations) / len(locations)
    avg_lng = sum(loc["lng"] for loc in locations) / len(locations)

    lats = [loc["lat"] for loc in locations]
    lngs = [loc["lng"] for loc in locations]
    spread = max(max(lats) - min(lats), max(lngs) - min(lngs))

    if spread < 0.05:
        zoom = 14
    elif spread < 0.15:
        zoom = 13
    elif spread < 0.5:
        zoom = 12
    elif spread < 1.5:
        zoom = 10
    elif spread < 5:
        zoom = 8
    else:
        zoom = 6

    m = folium.Map(location=[avg_lat, avg_lng], zoom_start=zoom,
                   tiles="CartoDB positron", control_scale=True)

    days_data: dict[int, list[dict]] = {}
    for loc in locations:
        days_data.setdefault(loc.get("day", 1), []).append(loc)

    slot_order = {"Morning": 0, "Afternoon": 1, "Evening": 2}

    for day_num in sorted(days_data.keys()):
        day_locs = days_data[day_num]
        day_locs.sort(key=lambda x: slot_order.get(x.get("time_slot", "Morning"), 0))
        color = DAY_COLORS[(day_num - 1) % len(DAY_COLORS)]

        if start_date:
            date_obj = start_date + timedelta(days=day_num - 1)
            day_label = f"Day {day_num} ({date_obj.strftime('%m/%d %a')})"
        else:
            day_label = f"Day {day_num}"

        fg = folium.FeatureGroup(name=day_label)
        coords_for_line = []

        for idx, loc in enumerate(day_locs):
            lat, lng = loc["lat"], loc["lng"]
            coords_for_line.append([lat, lng])

            slot = loc.get("time_slot", "")
            slot_emoji = {"Morning": "☀️", "Afternoon": "🌤️", "Evening": "🌙"}.get(slot, "📍")
            desc = loc.get("description", "")
            marker_number = idx + 1

            popup_html = f"""
            <div style="font-family: -apple-system, sans-serif; min-width: 180px; max-width: 240px;">
                <div style="background:{color}; color:white; padding:6px 10px;
                            border-radius:6px 6px 0 0; font-weight:700; font-size:12px;">
                    {day_label} · {slot_emoji} {slot}</div>
                <div style="padding:8px 10px;">
                    <div style="font-weight:700; font-size:13px; margin-bottom:4px;">{loc['name']}</div>
                    <div style="font-size:11px; color:#666;">{desc}</div>
                    <div style="font-size:10px; color:#999; margin-top:6px;">📍 {lat:.4f}, {lng:.4f}</div>
                </div>
            </div>
            """

            icon_html = f"""
            <div style="background:{color}; color:white; width:26px; height:26px;
                        border-radius:50%; display:flex; align-items:center; justify-content:center;
                        font-size:12px; font-weight:700; border:2px solid white;
                        box-shadow:0 2px 6px rgba(0,0,0,0.3);">{marker_number}</div>
            """

            folium.Marker(
                location=[lat, lng],
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{day_label} #{marker_number}: {loc['name']}",
                icon=folium.DivIcon(html=icon_html, icon_size=(26, 26), icon_anchor=(13, 13)),
            ).add_to(fg)

        if len(coords_for_line) >= 2:
            AntPath(
                locations=coords_for_line, color=color, weight=3,
                opacity=0.7, dash_array=[10, 20], delay=1000, pulse_color=color,
            ).add_to(fg)

        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Legend
    legend_items = ""
    for day_num in sorted(days_data.keys()):
        color = DAY_COLORS[(day_num - 1) % len(DAY_COLORS)]
        if start_date:
            date_obj = start_date + timedelta(days=day_num - 1)
            label = f"Day {day_num} ({date_obj.strftime('%m/%d')})"
        else:
            label = f"Day {day_num}"
        legend_items += f"""
            <div style="display:flex; align-items:center; margin-bottom:4px;">
                <div style="width:14px; height:14px; background:{color};
                            border-radius:50%; margin-right:8px; border:1px solid #ddd;"></div>
                <span style="font-size:12px;">{label}</span>
            </div>
        """
    legend_html = f"""
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000; background:white;
                padding:12px 16px; border-radius:8px; box-shadow:0 2px 10px rgba(0,0,0,0.15);
                font-family:-apple-system, sans-serif;">
        <div style="font-weight:700; font-size:13px; margin-bottom:8px;">🗺️ {city_name}</div>
        {legend_items}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def display_itinerary_map(itinerary: str, destination: str, num_days: int,
                          start_date: datetime, llm, key_suffix: str = ""):
    """Display one map per city using tabs. Single city = no tabs."""
    st.markdown("### 🗺️ Itinerary Map")
    st.caption("One map per city · Click markers for details · Toggle days with layer control")

    cache_key = f"map_locations_{destination}_{num_days}{key_suffix}"
    if cache_key not in st.session_state:
        with st.spinner("📍 Extracting locations and coordinates..."):
            st.session_state[cache_key] = extract_locations_from_itinerary(
                itinerary, destination, num_days, llm)
    locations = st.session_state[cache_key]

    if not locations:
        st.warning("🗺️ Could not extract map locations from this itinerary.")
        return

    # Group by city
    city_locs: dict[str, list] = {}
    for loc in locations:
        city_locs.setdefault(loc.get("city", destination), []).append(loc)

    if len(city_locs) == 1:
        # Single city — render directly
        city_name = list(city_locs.keys())[0]
        map_obj = create_city_map(city_locs[city_name], city_name, start_date)
        if map_obj:
            st_folium(map_obj, use_container_width=True, height=520, key=f"map_{key_suffix}_0")
    else:
        # Multiple cities — use tabs
        tabs = st.tabs([f"📍 {city}" for city in city_locs.keys()])
        for idx, (city_name, locs) in enumerate(city_locs.items()):
            with tabs[idx]:
                map_obj = create_city_map(locs, city_name, start_date)
                if map_obj:
                    st_folium(map_obj, use_container_width=True, height=520, key=f"map_{key_suffix}_{idx}")

    # Daily summary panel
    st.markdown("#### 📊 Locations by Day")
    days_data: dict[int, list[dict]] = {}
    for loc in locations:
        days_data.setdefault(loc["day"], []).append(loc)

    max_cols = min(len(days_data), 4)
    cols = st.columns(max_cols)

    for idx, day_num in enumerate(sorted(days_data.keys())):
        col = cols[idx % max_cols]
        day_locs = days_data[day_num]
        color = DAY_COLORS[(day_num - 1) % len(DAY_COLORS)]

        if start_date:
            date_obj = start_date + timedelta(days=day_num - 1)
            day_label = f"Day {day_num} ({date_obj.strftime('%m/%d')})"
        else:
            day_label = f"Day {day_num}"

        with col:
            items_html = "<br>".join(
                f"{'☀️' if l.get('time_slot')=='Morning' else '🌤️' if l.get('time_slot')=='Afternoon' else '🌙'} {l['name']}"
                for l in day_locs
            )
            st.markdown(f"""
            <div style="border-left:4px solid {color}; padding:10px 14px;
                        margin-bottom:10px; background:white; border-radius:0 12px 12px 0;
                        border:1px solid #F1F5F9; border-left:4px solid {color};
                        box-shadow:0 1px 4px rgba(0,0,0,0.04);">
                <div style="font-weight:700; font-size:0.88rem; margin-bottom:4px; color:#1E293B;">{day_label}</div>
                <div style="font-size:0.82rem; color:#64748B; line-height:1.6;">{items_html}</div>
            </div>
            """, unsafe_allow_html=True)


# ==================== PHOTO LANDMARK RECOGNITION ====================

def recognize_landmark_from_image(image_bytes: bytes, mime_type: str, llm) -> dict:
    """Use Gemini Vision to identify a landmark/attraction from a photo."""
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": """Identify the landmark, attraction, or notable place in this photo.

Return ONLY a JSON object with these exact keys:
{
    "name": "Official name of the place",
    "city": "City where it's located",
    "country": "Country",
    "type": "One of: temple, museum, park, landmark, restaurant, beach, market, church, palace, monument, neighborhood, nature, other",
    "description": "2-3 sentences about what makes this place special and worth visiting",
    "suggested_duration": "Recommended visit time, e.g. '1.5 hours', '2 hours', '30 minutes'",
    "best_time": "Morning or Afternoon or Evening",
    "tips": "1-2 practical tips for visiting (max 30 words)",
    "confidence": "high if you're very sure, medium if somewhat sure, low if guessing"
}

If you cannot identify any specific place, set name to "Unidentified" and confidence to "low".
Return ONLY the JSON object. No markdown, no backticks, no extra text.""",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"},
            },
        ]
    )

    try:
        response = llm.invoke([message])
        raw = response.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
        s, e = raw.find('{'), raw.rfind('}') + 1
        if s != -1 and e > s:
            result = json.loads(raw[s:e])
            for k, v in {"name": "Unidentified", "city": "Unknown", "country": "", "type": "landmark",
                         "description": "", "suggested_duration": "1 hour", "best_time": "Morning",
                         "tips": "", "confidence": "medium"}.items():
                result.setdefault(k, v)
            return result
    except Exception as ex:
        print(f"[Photo] Recognition failed: {ex}")

    return {
        "name": "Unidentified", "city": "Unknown", "country": "", "type": "other",
        "description": "Could not identify the place in this photo.",
        "suggested_duration": "1 hour", "best_time": "Morning", "tips": "", "confidence": "low",
    }


def integrate_landmarks_into_itinerary(current_itinerary: str, landmarks: list[dict],
                                        day_assignments: dict, destination: str,
                                        num_days: int, budget: float, llm) -> str:
    """Let LLM weave user-selected landmarks into the existing itinerary."""
    landmarks_desc = ""
    for lm in landmarks:
        assigned_day = day_assignments.get(lm["name"], "auto")
        landmarks_desc += f"""
- **{lm['name']}** ({lm['city']})
  Type: {lm['type']}
  Description: {lm['description']}
  Duration: {lm['suggested_duration']}
  Best time: {lm['best_time']}
  Tips: {lm['tips']}
  → Assign to: Day {assigned_day}
"""

    system_prompt = f"""You are revising a travel itinerary to incorporate new attractions the user wants to visit.

CURRENT ITINERARY:
{current_itinerary}

NEW ATTRACTIONS TO ADD:
{landmarks_desc}

RULES:
1. Insert each new attraction into the specified day
2. If day is "auto", choose the most logical day based on location proximity and schedule gaps
3. Adjust the existing schedule to make room — you may:
   - Shift activity times
   - Remove or shorten less important activities if the day is too packed
   - Reorder activities for better geographical flow
4. Keep the same format and structure as the original itinerary
5. Maintain budget of ${budget:.2f} total
6. Add practical transition notes (e.g., "15 min walk from..." or "Take subway line X")
7. Mark newly added activities with ✨ emoji so the user can spot them easily
8. Keep ALL other days that don't have new additions EXACTLY the same

Output the complete revised {num_days}-day itinerary."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please revise the itinerary to include the new attractions."),
    ])
    return response.content


TYPE_ICONS = {
    "temple": "⛩️", "museum": "🏛️", "park": "🌳", "landmark": "🏰",
    "restaurant": "🍽️", "beach": "🏖️", "market": "🛍️", "church": "⛪",
    "palace": "👑", "monument": "🗽", "neighborhood": "🏘️", "nature": "🌿", "other": "📍",
}
CONFIDENCE_BADGES = {
    "high": ("✅ High confidence", "#27ae60"),
    "medium": ("⚠️ Medium confidence", "#f39c12"),
    "low": ("❓ Low confidence", "#e74c3c"),
}


def display_photo_landmark_feature(current_itinerary: str, destination: str, num_days: int,
                                    budget: float, start_date: datetime, gemini_api_key: str):
    """Complete photo landmark recognition + itinerary integration UI."""
    st.markdown("### 📸 Add Places from Photos")
    st.caption("Upload photos of places you want to visit — AI will identify them and weave them into your itinerary.")

    uploaded_files = st.file_uploader(
        "Upload attraction photos (1-5 images)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="landmark_photos",
        help="Upload photos of landmarks, temples, restaurants, parks — anything you want to add to your trip!",
    )

    if not uploaded_files:
        return

    if len(uploaded_files) > 5:
        st.warning("⚠️ Maximum 5 photos at a time. Only the first 5 will be processed.")
        uploaded_files = uploaded_files[:5]

    # Identify button
    if st.button("🔍 Identify Places", type="primary", use_container_width=True, key="identify_btn"):
        llm_vision = ChatGoogleGenerativeAI(
            model=HELPER_MODEL, google_api_key=gemini_api_key, temperature=0.3)

        results = []
        with st.spinner(f"🔍 Analyzing {len(uploaded_files)} photo(s)..."):
            for f in uploaded_files:
                image_bytes = f.read()
                f.seek(0)
                result = recognize_landmark_from_image(image_bytes, f.type or "image/jpeg", llm_vision)
                result["file_name"] = f.name
                result["thumbnail_b64"] = base64.standard_b64encode(image_bytes).decode("utf-8")
                result["mime_type"] = f.type or "image/jpeg"
                results.append(result)

        st.session_state.photo_recognition_results = results

    # Show results
    if "photo_recognition_results" not in st.session_state or not st.session_state.photo_recognition_results:
        return

    results = st.session_state.photo_recognition_results
    st.markdown("#### 🎯 Identified Places")
    st.caption("Select which places to add to your itinerary and choose the day.")

    selected_landmarks = []

    for idx, result in enumerate(results):
        type_icon = TYPE_ICONS.get(result.get("type", "other"), "📍")
        conf = result.get("confidence", "medium")
        conf_label, conf_color = CONFIDENCE_BADGES.get(conf, CONFIDENCE_BADGES["medium"])

        col_img, col_info = st.columns([1, 2])

        with col_img:
            st.image(
                f"data:{result['mime_type']};base64,{result['thumbnail_b64']}",
                use_container_width=True,
            )

        with col_info:
            st.markdown(f"""
            <div style="background:#fff; border:1.5px solid #F1F5F9; border-radius:16px; padding:16px 18px; margin-bottom:12px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.04); transition:all 0.25s ease;">
                <div style="font-size:1.1rem; font-weight:700; color:#1E293B;">{type_icon} {result['name']}</div>
                <div style="font-size:.85rem; color:#64748B; margin-bottom:6px;">
                    📍 {result['city']}{(', ' + result['country']) if result.get('country') else ''}
                    <span style="display:inline-block; padding:3px 10px; border-radius:20px; font-size:.72rem; font-weight:600; background:{conf_color}18; color:{conf_color}; margin-left:8px;">
                        {conf_label}
                    </span>
                </div>
                <div style="font-size:.88rem; color:#475569; line-height:1.5; margin-bottom:8px;">{result['description']}</div>
                <div style="font-size:.82rem; color:#A78BFA;">
                    ⏱️ {result['suggested_duration']} · 🕐 Best: {result['best_time']} · 💡 {result.get('tips', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_check, col_day = st.columns([1, 1])
            with col_check:
                include = st.checkbox(
                    "Add to itinerary", key=f"include_landmark_{idx}",
                    value=(result["confidence"] != "low" and result["name"] != "Unidentified"),
                )
            with col_day:
                if include:
                    day_options = {}
                    for d in range(1, num_days + 1):
                        if start_date:
                            date_obj = start_date + timedelta(days=d - 1)
                            day_options[f"Day {d} ({date_obj.strftime('%m/%d %a')})"] = d
                        else:
                            day_options[f"Day {d}"] = d
                    day_options["Auto (best fit)"] = "auto"

                    selected_day_label = st.selectbox(
                        "Which day?", options=list(day_options.keys()), key=f"day_select_{idx}")
                    selected_landmarks.append({
                        **result,
                        "assigned_day": day_options[selected_day_label],
                    })

        st.markdown("---")

    # Confirm
    if selected_landmarks:
        st.markdown(f"**{len(selected_landmarks)} place(s) selected**")

        if st.button("✨ Add to My Itinerary", type="primary", use_container_width=True, key="add_landmarks_btn"):
            llm_planner = ChatGoogleGenerativeAI(
                model=MAIN_MODEL, google_api_key=gemini_api_key, temperature=0.7)

            day_assignments = {lm["name"]: lm["assigned_day"] for lm in selected_landmarks}

            with st.spinner("✨ Integrating new places into your itinerary..."):
                revised_itinerary = integrate_landmarks_into_itinerary(
                    current_itinerary=current_itinerary,
                    landmarks=selected_landmarks,
                    day_assignments=day_assignments,
                    destination=destination,
                    num_days=num_days,
                    budget=budget,
                    llm=llm_planner,
                )

            st.session_state.itinerary = revised_itinerary
            st.session_state.photo_recognition_results = []
            # Clear map cache
            keys_to_remove = [k for k in st.session_state if k.startswith("map_locations_")]
            for k in keys_to_remove:
                del st.session_state[k]

            st.success(f"✅ Added {len(selected_landmarks)} place(s) to your itinerary! (Look for ✨ markers)")
            st.rerun()
    else:
        st.info("👆 Select at least one place to add to your itinerary.")


# ==================== XIAOHONGSHU FACT-CHECKER ====================

def extract_claims_from_post(post_text: str, llm) -> list[dict]:
    """
    从小红书文案中提取可验证的信息点。
    返回结构化的 claim 列表，每条包含中英文、分类、搜索关键词等。
    """
    prompt = f"""You are a bilingual (Chinese-English) travel content analyst.

From the Xiaohongshu (小红书) travel post below, extract ALL verifiable claims — things that can be checked against English-language sources.

For EACH claim, provide:
- claim_zh: the original Chinese claim (quote or close paraphrase)
- claim_en: English translation of the claim
- category: one of "restaurant", "attraction", "tip", "fact", "product", "accommodation"
- entity_name: the specific place/business name in English (romanized if needed)
- entity_name_zh: the Chinese name
- location: city and country
- search_query: a good English search query to verify this claim (4-8 words)
- importance: "high" if it's a key recommendation, "medium" if supporting detail, "low" if minor

Focus on:
- Restaurant/cafe recommendations and specific dishes
- Attraction claims ("must visit", "hidden gem", "best X in Y")
- Specific prices, hours, or practical tips
- Rankings or superlatives ("number 1", "best", "most famous")
- Claims about popularity or reputation

Return ONLY a JSON array. No markdown, no backticks, no extra text.

XIAOHONGSHU POST:
{post_text[:3000]}
"""
    try:
        response = llm.invoke([
            SystemMessage(content="You are a bilingual content analyst. Return ONLY a raw JSON array."),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
        start, end = raw.find('['), raw.rfind(']') + 1
        if start != -1 and end > start:
            claims = json.loads(raw[start:end])
            valid = []
            for c in claims:
                if "claim_zh" in c and "search_query" in c:
                    c.setdefault("claim_en", "")
                    c.setdefault("category", "tip")
                    c.setdefault("entity_name", "")
                    c.setdefault("entity_name_zh", "")
                    c.setdefault("location", "")
                    c.setdefault("importance", "medium")
                    valid.append(c)
            return valid
    except Exception as e:
        print(f"[FactCheck] Claim extraction failed: {e}")
    return []


def verify_claim_against_english_sources(claim: dict, serp_wrapper_instance, llm) -> dict:  # serp_wrapper_instance kept for signature compat
    """
    对单条声称进行英文源校验:
    1. 用 SerpAPI 搜索英文信息
    2. 让 LLM 分析搜索结果，判断是否佐证该声称
    """
    search_query = claim.get("search_query", "")
    entity = claim.get("entity_name", "")

    # Do 1-2 searches with different angles
    search_results = []
    queries = [search_query]
    if entity and entity.lower() not in search_query.lower():
        queries.append(f"{entity} {claim.get('location', '')} review")

    for q in queries[:2]:
        try:
            # result = serp_wrapper_instance.run(q)  # SerpAPI
            result = tavily_client.search(q, max_results=3)
            text = "\n".join([r.get("content", "")[:300] for r in result.get("results", [])])
            search_results.append(f"Query: {q}\n{text}")
        except Exception as e:
            search_results.append(f"Query: {q}\nSearch failed: {e}")

    combined_results = "\n\n---\n\n".join(search_results)

    analysis_prompt = f"""You are a fact-checker comparing a Chinese social media (Xiaohongshu) travel claim against English-language sources.

CLAIM FROM XIAOHONGSHU:
- Chinese: {claim.get('claim_zh', '')}
- English: {claim.get('claim_en', '')}
- Entity: {claim.get('entity_name', '')} ({claim.get('entity_name_zh', '')})
- Location: {claim.get('location', '')}
- Category: {claim.get('category', '')}

ENGLISH SEARCH RESULTS:
{combined_results[:4000]}

Analyze whether English-language sources support this claim. Return ONLY a JSON object:
{{
    "verification_status": "confirmed | partially_confirmed | unverified | contradicted",
    "confidence_score": 0.85,
    "english_evidence": "2-3 sentence summary of what English sources say (or 'No relevant English sources found')",
    "english_sources": ["source names found, e.g. TripAdvisor, Lonely Planet, Reddit"],
    "discrepancies": "Any differences between the Xiaohongshu claim and English sources (or 'None')",
    "verdict": "One sentence: is this a legit recommendation, overhyped, or unverifiable?"
}}

Guidelines for verification_status:
- "confirmed": English sources clearly support the same claim
- "partially_confirmed": English sources mention the entity but not the specific claim, or have mixed reviews
- "unverified": Cannot find relevant English sources (could be a local/niche place unknown outside China)
- "contradicted": English sources directly contradict the claim

Guidelines for confidence_score:
- 0.9-1.0: Multiple reputable English sources confirm
- 0.7-0.89: Some English mentions, generally supportive
- 0.5-0.69: Limited or mixed English evidence
- 0.3-0.49: Very little English presence, hard to verify
- 0.0-0.29: No English evidence OR contradicted

Return ONLY the JSON object. No markdown."""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a bilingual fact-checker. Return ONLY a raw JSON object."),
            HumanMessage(content=analysis_prompt),
        ])
        raw = response.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
        s, e = raw.find('{'), raw.rfind('}') + 1
        if s != -1 and e > s:
            result = json.loads(raw[s:e])
            result.setdefault("verification_status", "unverified")
            result.setdefault("confidence_score", 0.5)
            result.setdefault("english_evidence", "Analysis unavailable")
            result.setdefault("english_sources", [])
            result.setdefault("discrepancies", "None")
            result.setdefault("verdict", "Unable to determine")
            return {**claim, **result}
    except Exception as ex:
        print(f"[FactCheck] Verification failed for '{claim.get('entity_name', '')}': {ex}")

    return {
        **claim,
        "verification_status": "unverified",
        "confidence_score": 0.0,
        "english_evidence": "Verification failed due to an error.",
        "english_sources": [],
        "discrepancies": "N/A",
        "verdict": "Could not verify — please check manually.",
    }


FC_STATUS_CONFIG = {
    "confirmed": {"icon": "✅", "color": "#27ae60", "label": "外网验证通过", "bg": "#eafaf1"},
    "partially_confirmed": {"icon": "🟡", "color": "#f39c12", "label": "外网部分提及", "bg": "#fef9e7"},
    "unverified": {"icon": "❓", "color": "#95a5a6", "label": "外网无相关信息", "bg": "#f2f3f4"},
    "contradicted": {"icon": "❌", "color": "#e74c3c", "label": "外网信息矛盾", "bg": "#fdedec"},
}

FC_CATEGORY_ICONS = {
    "restaurant": "🍽️", "attraction": "🏛️", "tip": "💡",
    "fact": "📊", "product": "🛍️", "accommodation": "🏨",
}


def render_confidence_bar(score: float) -> str:
    """Generate HTML for a confidence score bar."""
    percentage = int(score * 100)
    if score >= 0.7:
        bar_color = "#27ae60"
    elif score >= 0.5:
        bar_color = "#f39c12"
    elif score >= 0.3:
        bar_color = "#e67e22"
    else:
        bar_color = "#e74c3c"

    return f"""
    <div style="display:flex; align-items:center; gap:8px;">
        <div style="flex:1; background:#eee; border-radius:4px; height:8px; overflow:hidden;">
            <div style="width:{percentage}%; background:{bar_color}; height:100%; border-radius:4px;"></div>
        </div>
        <span style="font-size:0.85rem; font-weight:700; color:{bar_color};">{percentage}%</span>
    </div>
    """


def display_fact_check_feature(serp_wrapper_instance, gemini_api_key: str):
    """完整的小红书文案校验 UI 组件。"""
    st.markdown("### 🔍 rednote Content Fact-Checker")
    st.caption("Paste a rednote (小红书) travel post — AI extracts key claims and cross-checks them against English sources. Find out if it's influencer hype or genuinely well-reviewed!")

    st.markdown("""
    <style>
    .fc-card {
        background: #ffffff;
        border: 1.5px solid #F1F5F9;
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.25s ease;
    }
    .fc-card:hover {
        border-color: #A78BFA;
        box-shadow: 0 6px 20px rgba(167,139,250,0.12);
        transform: translateY(-2px);
    }
    .fc-claim-zh { font-size: 0.95rem; color: #334155; line-height: 1.5; margin-bottom: 6px; }
    .fc-claim-en { font-size: 0.85rem; color: #94A3B8; font-style: italic; margin-bottom: 10px; }
    .fc-entity { font-size: 1.05rem; font-weight: 700; color: #1E293B; margin-bottom: 4px; }
    .fc-evidence { font-size: 0.88rem; color: #475569; line-height: 1.5; margin: 8px 0; }
    .fc-sources { font-size: 0.82rem; color: #64748B; }
    .fc-verdict { font-size: 0.9rem; font-weight: 600; margin-top: 8px; padding: 10px 14px; border-radius: 12px; }
    .fc-discrepancy { font-size: 0.85rem; color: #EF4444; margin-top: 6px; }
    .fc-status-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .fc-summary-card {
        background: linear-gradient(135deg, #FF6B8A 0%, #A78BFA 100%);
        color: white;
        padding: 22px 26px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(255,107,138,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    # Input area
    post_text = st.text_area(
        "📱 Paste rednote Post",
        placeholder="Paste a rednote travel recommendation here...\n\nExample:\nMust-eat in Tokyo! This hidden ramen shop in Shinjuku, Fuunji, is the best ramen I've ever had!\nWorth the 1-hour queue, rich fish-pork bone broth with homemade noodles...\nGoogle rating 4.8, even locals line up...",
        height=200,
        key="xiaohongshu_post_input",
    )

    if not post_text:
        return

    if st.button("🔍 Start Fact-Check", type="primary", use_container_width=True,
                 key="start_fact_check"):

        llm = ChatGoogleGenerativeAI(
            model=MAIN_MODEL, google_api_key=gemini_api_key, temperature=0.3)

        # Step 1: Extract claims
        with st.spinner("📝 Extracting key claims from the post..."):
            claims = extract_claims_from_post(post_text, llm)

        if not claims:
            st.warning(
                "⚠️ Could not extract verifiable claims. Make sure the post contains specific recommendations (restaurant names, attractions, etc.).")
            return

        st.info(f"📋 Extracted **{len(claims)}** verifiable claims. Searching English sources one by one...")

        # Step 2: Verify each claim
        verified_claims = []
        progress_bar = st.progress(0, text="Searching English sources...")

        for idx, claim in enumerate(claims):
            progress_bar.progress(
                (idx + 1) / len(claims),
                text=f"🔍 [{idx + 1}/{len(claims)}] Checking: {claim.get('entity_name', claim.get('claim_en', '')[:30])}..."
            )
            verified = verify_claim_against_english_sources(claim, serp_wrapper_instance, llm)
            verified_claims.append(verified)

        progress_bar.empty()

        # Save to session
        st.session_state.fact_check_results = verified_claims

        # ── Display Results ──────────────────────────────────────
    if "fact_check_results" not in st.session_state or not st.session_state.fact_check_results:
        return

    results = st.session_state.fact_check_results

    # Summary card
    total = len(results)
    confirmed = sum(1 for r in results if r["verification_status"] == "confirmed")
    partial = sum(1 for r in results if r["verification_status"] == "partially_confirmed")
    unverified_count = sum(1 for r in results if r["verification_status"] == "unverified")
    contradicted = sum(1 for r in results if r["verification_status"] == "contradicted")
    avg_confidence = sum(r.get("confidence_score", 0) for r in results) / total if total else 0

    if avg_confidence >= 0.7:
        overall_note = "👍 Most claims are backed by English sources — high credibility"
    elif avg_confidence >= 0.4:
        overall_note = "⚠️ Some claims verified, some not — use with caution"
    else:
        overall_note = "🚨 Very little English coverage — may be a rednote-only recommendation"

    st.markdown(f"""
        <div class="fc-summary-card">
            <div style="font-size:1.2rem; font-weight:700; margin-bottom:10px;">📊 Verification Report</div>
            <div style="display:flex; gap:20px; flex-wrap:wrap;">
                <div>📝 Claims: <b>{total}</b></div>
                <div>✅ Confirmed: <b>{confirmed}</b></div>
                <div>🟡 Partial: <b>{partial}</b></div>
                <div>❓ Unverified: <b>{unverified_count}</b></div>
                <div>❌ Contradicted: <b>{contradicted}</b></div>
            </div>
            <div style="margin-top:10px; font-size:0.95rem;">
                Overall Confidence: <b>{avg_confidence * 100:.0f}%</b> — {overall_note}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Individual claim cards
    st.markdown("#### 📋 Claim-by-Claim Results")

    for idx, result in enumerate(results):
        status = result.get("verification_status", "unverified")
        cfg = FC_STATUS_CONFIG.get(status, FC_STATUS_CONFIG["unverified"])
        cat_icon = FC_CATEGORY_ICONS.get(result.get("category", "tip"), "📍")
        confidence = result.get("confidence_score", 0)
        sources = result.get("english_sources", [])

        entity_display = result.get('entity_name', '')
        if result.get('entity_name_zh'):
            entity_display += f" ({result['entity_name_zh']})"

        sources_html = f'<div class="fc-sources">📰 Sources: {", ".join(sources)}</div>' if sources else ''

        discrepancy_html = ''
        if result.get("discrepancies") and result["discrepancies"] not in ("None", "N/A", ""):
            discrepancy_html = f'<div class="fc-discrepancy">⚠️ Discrepancy: {result["discrepancies"]}</div>'

        st.markdown(f"""
            <div class="fc-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <div class="fc-entity">{cat_icon} {entity_display}</div>
                    <span class="fc-status-badge" style="background:{cfg['bg']}; color:{cfg['color']};">
                        {cfg['icon']} {cfg['label']}
                    </span>
                </div>
                <div class="fc-claim-zh">📱 "{result.get('claim_zh', '')}"</div>
                <div class="fc-claim-en">🌐 {result.get('claim_en', '')}</div>
                <div style="margin:10px 0;">
                    <div style="font-size:0.82rem; color:#888; margin-bottom:4px;">Confidence</div>
                    {render_confidence_bar(confidence)}
                </div>
                <div class="fc-evidence">
                    <b>🔎 English Evidence:</b> {result.get('english_evidence', 'N/A')}
                </div>
                {sources_html}
                {discrepancy_html}
                <div class="fc-verdict" style="background:{cfg['bg']};">
                    💬 {result.get('verdict', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Overall advice
    st.markdown("---")
    if avg_confidence >= 0.7:
        st.success(
            "✅ **Overall Assessment**: The recommendations in this post are well-supported by English sources. Feel free to follow them, but double-check hours and prices before you go.")
    elif avg_confidence >= 0.4:
        st.warning(
            "⚠️ **Overall Assessment**: Some claims are backed by English sources, others could not be verified. Cross-reference with Google Maps ratings and TripAdvisor before deciding.")
    else:
        st.error(
            "🚨 **Overall Assessment**: Almost no English coverage found. This could be a niche recommendation for Chinese tourists, or the information may be inaccurate. Check Google Maps reviews and photos before committing.")

    # Clear button
    if st.button("🗑️ Clear Results", use_container_width=True, key="clear_fact_check"):
        st.session_state.fact_check_results = []
        st.rerun()

# ==================== Audio Tour 语音导览 ====================

AUDIO_TOUR_DIR = Path(__file__).parent / "audio_tours"
AUDIO_TOUR_DIR.mkdir(exist_ok=True)

AUDIO_ARCHITECTURE_PROMPT = """你是一位专业的建筑导览解说员。请根据以下地点信息，用中文撰写一段生动的建筑导览词。

要求：
1. 描述建筑风格、设计元素、城市规划特色
2. 平衡专业性与通俗性，让普通游客也能听懂
3. 用口语化的方式表达，就像导游在旁边讲解一样
4. 不要加任何标题、格式标记或链接
5. 内容要连贯自然，适合直接转为语音
6. 字数控制在 {word_limit} 字左右

地点：{location}
当天行程参考：{day_context}
"""

AUDIO_HISTORY_PROMPT = """你是一位资深的历史导览解说员。请根据以下地点信息，用中文撰写一段引人入胜的历史导览词。

要求：
1. 讲述与该地点相关的历史事件、人物和故事
2. 包含一些鲜为人知的有趣历史细节
3. 用叙事的方式讲述，像在讲故事一样生动
4. 不要加任何标题、格式标记或链接
5. 内容要连贯自然，适合直接转为语音
6. 字数控制在 {word_limit} 字左右

地点：{location}
当天行程参考：{day_context}
"""

AUDIO_CULTURE_PROMPT = """你是一位热情的文化导览解说员。请根据以下地点信息，用中文撰写一段关于当地文化的导览词。

要求：
1. 介绍当地的传统习俗、艺术、音乐和文化活动
2. 解释文化背后的意义，帮助游客理解当地生活方式
3. 用温暖亲切的语气表达
4. 不要加任何标题、格式标记或链接
5. 内容要连贯自然，适合直接转为语音
6. 字数控制在 {word_limit} 字左右

地点：{location}
当天行程参考：{day_context}
"""

AUDIO_CULINARY_PROMPT = """你是一位美食导览解说员。请根据以下地点信息，用中文撰写一段关于当地美食的导览词。

要求：
1. 介绍当地特色美食、餐厅、市场和饮食文化
2. 描述食物的味道、做法和文化意义
3. 用热情洋溢的语气，让人听了就想去尝尝
4. 不要加任何标题、格式标记或链接
5. 内容要连贯自然，适合直接转为语音
6. 字数控制在 {word_limit} 字左右

地点：{location}
当天行程参考：{day_context}
"""

AUDIO_ORCHESTRATOR_PROMPT = """你是一位专业的语音导览编排师。你的任务是将四段导览内容（建筑、历史、文化、美食）整合成一段连贯的中文语音导览。

要求：
1. 添加一段简短温暖的开场白，欢迎听众
2. 按照 建筑 → 历史 → 文化 → 美食 的顺序编排
3. 在每个主题之间添加自然的过渡语句，将主题串联起来
4. 添加一段简短的结尾，总结亮点并鼓励听众继续探索
5. 整体听起来像一位友好的导游在带你漫步
6. 不要加任何标题、格式标记或链接
7. 最终内容必须是纯文本，可以直接朗读

地点：{location}

建筑内容：
{architecture}

历史内容：
{history}

文化内容：
{culture}

美食内容：
{culinary}
"""


def generate_audio_tour_text(location: str, day_context: str, llm, duration_minutes: int = 5) -> str:
    """用 4 个主题 prompt 分别生成，再用 orchestrator 拼接成完整导览词。"""
    # 按语速估算字数：中文朗读约 250 字/分钟
    total_words = duration_minutes * 250
    word_per_section = total_words // 4

    prompts = {
        "architecture": AUDIO_ARCHITECTURE_PROMPT,
        "history": AUDIO_HISTORY_PROMPT,
        "culture": AUDIO_CULTURE_PROMPT,
        "culinary": AUDIO_CULINARY_PROMPT,
    }

    sections = {}
    for key, prompt_tpl in prompts.items():
        prompt = prompt_tpl.format(
            word_limit=word_per_section,
            location=location,
            day_context=day_context,
        )
        response = llm.invoke([
            SystemMessage(content="你是一位专业的中文旅游导览内容创作者。只输出导览正文，不要加任何格式标记。"),
            HumanMessage(content=prompt),
        ])
        sections[key] = response.content.strip()

    # Orchestrator 拼接
    orchestrator_prompt = AUDIO_ORCHESTRATOR_PROMPT.format(
        location=location,
        architecture=sections["architecture"],
        history=sections["history"],
        culture=sections["culture"],
        culinary=sections["culinary"],
    )
    response = llm.invoke([
        SystemMessage(content="你是一位中文语音导览编排师。输出纯文本导览词，不要加任何格式标记。"),
        HumanMessage(content=orchestrator_prompt),
    ])
    return response.content.strip()


def _split_text_for_tts(text: str, max_chars: int = 900) -> list[str]:
    """将长文本按句子边界切分，每段不超过 max_chars 字。"""
    separators = ["。", "！", "？", "；", ".", "!", "?", "\n"]
    chunks = []
    current = ""
    for char in text:
        current += char
        if len(current) >= max_chars and char in separators:
            chunks.append(current.strip())
            current = ""
    if current.strip():
        chunks.append(current.strip())
    return chunks


def text_to_speech_step(text: str, output_path: str, api_key: str, voice: str = "cixingnansheng") -> str:
    """使用 StepFun Step-TTS-2 将中文文本转为语音，自动分段合成。"""
    import subprocess
    client = StepOpenAI(api_key=api_key, base_url="https://api.stepfun.com/v1")
    chunks = _split_text_for_tts(text)

    if len(chunks) == 1:
        response = client.audio.speech.create(
            model="step-tts-2", voice=voice, input=chunks[0],
        )
        response.stream_to_file(output_path)
        return output_path

    # 多段：分别生成临时文件再拼接
    temp_files = []
    output_dir = Path(output_path).parent
    for i, chunk in enumerate(chunks):
        temp_path = str(output_dir / f"_temp_chunk_{i}.mp3")
        response = client.audio.speech.create(
            model="step-tts-2", voice=voice, input=chunk,
        )
        response.stream_to_file(temp_path)
        temp_files.append(temp_path)

    # 用 ffmpeg 拼接（如果可用），否则二进制拼接
    try:
        list_file = str(output_dir / "_temp_filelist.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for tf in temp_files:
                f.write(f"file '{tf}'\n")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", output_path],
            capture_output=True, check=True,
        )
        Path(list_file).unlink(missing_ok=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg 不可用，直接二进制拼接 MP3
        with open(output_path, "wb") as out_f:
            for tf in temp_files:
                with open(tf, "rb") as in_f:
                    out_f.write(in_f.read())

    # 清理临时文件
    for tf in temp_files:
        Path(tf).unlink(missing_ok=True)

    return output_path


def extract_day_content_from_itinerary(itinerary: str, day_num: int) -> str:
    """从行程中提取某一天的内容。支持 Day 1 / ## Day 1 / **Day 1:** / 第1天 / 第一天 等格式。"""
    cn_nums = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七",
               8: "八", 9: "九", 10: "十"}
    cn_num = cn_nums.get(day_num, str(day_num))
    # 匹配当前天：Day 1 / **Day 1** / ## Day 1 / 第1天 / 第一天
    day_header = rf'(?:Day\s*{day_num}|第\s*{day_num}\s*天|第\s*{cn_num}\s*天)'
    # 匹配下一天
    next_header = r'(?:Day\s*\d+|第\s*\d+\s*天|第\s*[一二三四五六七八九十]+\s*天)'
    # 允许 ## / ** / ### ** 等前缀
    prefix = r'(?:#{1,3}\s*)?(?:\*{1,2})?'
    day_pattern = re.compile(
        r'(?:^|\n)\s*' + prefix + day_header + r'[^\n]*\n(.*?)(?=\n\s*' + prefix + next_header + r'|\Z)',
        re.DOTALL | re.IGNORECASE
    )
    match = day_pattern.search(itinerary)
    if match:
        return match.group(1).strip()
    return ""


def extract_city_for_day(itinerary: str, day_num: int, destination: str, llm) -> str:
    """用 LLM 判断某一天所在的城市。"""
    day_content = extract_day_content_from_itinerary(itinerary, day_num)
    if not day_content:
        return destination
    prompt = f"以下是旅行行程中 Day {day_num} 的内容，请只回答这天所在的城市名称（中文），不要其他文字：\n\n{day_content[:500]}"
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception:
        return destination


def display_audio_tour_feature(itinerary: str, destination: str, num_days: int,
                                start_date: datetime, gemini_api_key: str,
                                stepfun_api_key: str = ""):
    """语音导览 Streamlit UI。"""
    st.markdown("### 🎧 语音导览")
    st.caption("AI 为你的每一天行程生成专属中文语音导览，边走边听")

    if not stepfun_api_key:
        st.warning("⚠️ 请在侧边栏输入 StepFun API Key 以使用语音合成功能")
        return

    llm = ChatGoogleGenerativeAI(
        model=MAIN_MODEL, google_api_key=gemini_api_key, temperature=0.7)
    llm_helper = ChatGoogleGenerativeAI(
        model=HELPER_MODEL, google_api_key=gemini_api_key, temperature=0.3)

    # 提取城市列表
    if "audio_tour_cities" not in st.session_state:
        st.session_state.audio_tour_cities = extract_cities_from_itinerary(
            itinerary, destination, llm_helper)
    cities = st.session_state.audio_tour_cities

    # 语音选择 (Step-TTS-2)
    voice_options = {
        "磁性男声": "cixingnansheng",
        "温柔女声": "wenrounvsheng",
        "播音男声": "boyinnansheng",
        "知性姐姐": "zhixingjiejie",
        "儒雅男士": "ruyananshi",
        "亲切女声": "qinqienvsheng",
        "自信男声": "zixinnansheng",
        "优雅女声": "youyanvsheng",
        "气质温婉": "elegantgentle-female",
        "活力轻快": "livelybreezy-female",
        "深沉男音": "shenchennanyin",
        "甜美女声": "tianmeinvsheng",
    }
    selected_voice_label = st.selectbox("🎙️ 选择语音", list(voice_options.keys()), key="audio_tour_voice")
    selected_voice = voice_options[selected_voice_label]

    duration = st.slider("⏱️ 每段导览时长（分钟）", min_value=3, max_value=15, value=5,
                          step=1, key="audio_tour_duration")

    # 如果行程内容变了，或缓存天数不足，清除旧缓存重新提取
    _itin_hash = hash(itinerary[:500])
    _cached_days = st.session_state.get("audio_tour_all_day_contents", {})
    _need_refresh = (
        st.session_state.get("_audio_tour_itin_hash") != _itin_hash
        or len(_cached_days) < num_days * 0.5  # 缓存天数不到一半，说明之前解析有问题
    )
    if _need_refresh:
        for stale_key in ["audio_tour_all_day_contents", "audio_tour_day_city_map", "audio_tour_cities"]:
            st.session_state.pop(stale_key, None)
        st.session_state["_audio_tour_itin_hash"] = _itin_hash
        # 重新提取城市列表
        st.session_state.audio_tour_cities = extract_cities_from_itinerary(
            itinerary, destination, llm_helper)
        cities = st.session_state.audio_tour_cities

    # 预提取所有天的内容
    all_day_contents_key = "audio_tour_all_day_contents"
    if all_day_contents_key not in st.session_state:
        all_day_contents = {}
        for d in range(1, num_days + 1):
            content = extract_day_content_from_itinerary(itinerary, d)
            if content:
                all_day_contents[d] = content
        # fallback：如果正则全部失败，用 LLM 分割
        if not all_day_contents and itinerary.strip():
            split_prompt = (
                f"以下是一个 {num_days} 天的旅行行程。请把每一天的内容分别提取出来。\n"
                f"返回 JSON 对象，格式如 {{\"1\": \"第一天的完整内容...\", \"2\": \"第二天的完整内容...\"}}。\n"
                f"不要其他文字，只返回 JSON。\n\n{itinerary[:6000]}"
            )
            try:
                resp = llm_helper.invoke([HumanMessage(content=split_prompt)])
                raw = resp.content.strip()
                raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
                raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
                s, e = raw.find('{'), raw.rfind('}') + 1
                if s != -1 and e > s:
                    parsed = json.loads(raw[s:e])
                    all_day_contents = {int(k): v for k, v in parsed.items() if v}
            except Exception:
                pass
        # 最后的 fallback：按行数平均分割
        if not all_day_contents and itinerary.strip():
            lines = [l for l in itinerary.strip().split('\n') if l.strip()]
            lines_per_day = max(1, len(lines) // num_days)
            for d in range(1, num_days + 1):
                start_idx = (d - 1) * lines_per_day
                end_idx = d * lines_per_day if d < num_days else len(lines)
                chunk = '\n'.join(lines[start_idx:end_idx])
                if chunk.strip():
                    all_day_contents[d] = chunk
        st.session_state[all_day_contents_key] = all_day_contents
    all_day_contents = st.session_state[all_day_contents_key]

    # 用 LLM 判断每天属于哪个城市
    day_city_map_key = "audio_tour_day_city_map"
    if day_city_map_key not in st.session_state:
        if len(cities) > 1 and all_day_contents:
            prompt = (
                f"以下是一个旅行行程，涉及这些城市：{', '.join(cities)}。\n"
                f"请判断每一天主要在哪个城市。城市名必须使用以下原始名称之一：{', '.join(cities)}。\n"
                f"只返回 JSON 对象，格式如 {{\"1\": \"{cities[0]}\", \"2\": \"{cities[0]}\", \"3\": \"{cities[-1]}\"}}，不要其他文字。\n\n"
            )
            for d, content in all_day_contents.items():
                content_str = content if isinstance(content, str) else str(content)
                prompt += f"Day {d}:\n{content_str[:300]}\n\n"
            try:
                resp = llm_helper.invoke([HumanMessage(content=prompt)])
                raw = resp.content.strip()
                raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
                raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
                s, e = raw.find('{'), raw.rfind('}') + 1
                if s != -1 and e > s:
                    st.session_state[day_city_map_key] = json.loads(raw[s:e])
                else:
                    st.session_state[day_city_map_key] = {str(d): cities[0] for d in all_day_contents}
            except Exception:
                st.session_state[day_city_map_key] = {str(d): cities[0] for d in all_day_contents}
        else:
            # 单城市：所有天都属于该城市
            st.session_state[day_city_map_key] = {str(d): cities[0] for d in all_day_contents}
    day_city_map = st.session_state[day_city_map_key]

    # ── 城市总览 ──
    for city in cities:
        st.markdown(f"#### 📍 {city} 总览")
        overview_key = f"audio_tour_overview_{city}"
        overview_audio_key = f"audio_tour_overview_audio_{city}"

        if st.button(f"🎧 生成 {city} 总览导览", key=f"gen_overview_{city}"):
            with st.spinner(f"正在为 {city} 生成总览导览词..."):
                # 收集该城市所有天的内容
                city_context = ""
                for d in range(1, num_days + 1):
                    day_content = extract_day_content_from_itinerary(itinerary, d)
                    if city.lower() in day_content.lower() or city in day_content:
                        city_context += f"Day {d}:\n{day_content}\n\n"
                if not city_context:
                    city_context = itinerary[:2000]

                text = generate_audio_tour_text(city, city_context, llm, duration)
                st.session_state[overview_key] = text

            with st.spinner("正在生成语音..."):
                audio_path = str(AUDIO_TOUR_DIR / f"overview_{city}.mp3")
                text_to_speech_step(st.session_state[overview_key], audio_path, stepfun_api_key, selected_voice)
                st.session_state[overview_audio_key] = audio_path

        if overview_key in st.session_state:
            with st.expander("📝 导览文稿", expanded=False):
                st.markdown(st.session_state[overview_key])
        if overview_audio_key in st.session_state:
            st.audio(st.session_state[overview_audio_key], format="audio/mp3")
            with open(st.session_state[overview_audio_key], "rb") as f:
                st.download_button(f"📥 下载 {city} 总览", data=f, file_name=f"{city}_overview.mp3",
                                   mime="audio/mp3", key=f"dl_overview_{city}")

        # ── 按天生成 ──
        for day in range(1, num_days + 1):
            day_content = all_day_contents.get(day, "")
            if not day_content:
                continue
            # 用 LLM 映射结果判断这天属于哪个城市（模糊匹配）
            day_city_raw = day_city_map.get(str(day), cities[0])
            # 匹配逻辑：day_city_raw 或 city 互相包含即算匹配
            day_belongs_to_city = (
                day_city_raw == city
                or day_city_raw.lower() == city.lower()
                or day_city_raw in city or city in day_city_raw
                or day_city_raw.lower() in city.lower() or city.lower() in day_city_raw.lower()
            )
            if len(cities) > 1 and not day_belongs_to_city:
                continue

            if start_date:
                date_obj = start_date + timedelta(days=day - 1)
                day_label = f"Day {day} ({date_obj.strftime('%m/%d %a')})"
            else:
                day_label = f"Day {day}"

            # 提取当天亮点作为副标题
            first_line = day_content.split('\n')[0].strip()
            first_line = re.sub(r'^[-•*\d:.\s]+', '', first_line)[:30]
            st.markdown(f"**{day_label}** — {first_line}")

            day_text_key = f"audio_tour_day_{city}_{day}"
            day_audio_key = f"audio_tour_day_audio_{city}_{day}"

            col_btn, col_play = st.columns([1, 2])
            with col_btn:
                if st.button(f"🎧 生成", key=f"gen_day_{city}_{day}"):
                    with st.spinner(f"正在生成 {day_label} 导览..."):
                        # 提取该天的地点
                        day_location = f"{city} - {first_line}" if first_line else city
                        text = generate_audio_tour_text(day_location, day_content, llm, duration)
                        st.session_state[day_text_key] = text

                    with st.spinner("正在生成语音..."):
                        audio_path = str(AUDIO_TOUR_DIR / f"day_{city}_{day}.mp3")
                        text_to_speech_step(st.session_state[day_text_key], audio_path, stepfun_api_key, selected_voice)
                        st.session_state[day_audio_key] = audio_path

            with col_play:
                if day_audio_key in st.session_state:
                    st.audio(st.session_state[day_audio_key], format="audio/mp3")

            if day_text_key in st.session_state:
                with st.expander(f"📝 {day_label} 导览文稿", expanded=False):
                    st.markdown(st.session_state[day_text_key])
            if day_audio_key in st.session_state:
                with open(st.session_state[day_audio_key], "rb") as f:
                    st.download_button(f"📥 下载 {day_label}", data=f, file_name=f"day_{city}_{day}_tour.mp3",
                                       mime="audio/mp3", key=f"dl_day_{city}_{day}")

        st.markdown("---")


# ==================== Agent Nodes ====================
def researcher_agent(state: AgentState) -> AgentState:
    destination = state["destination"]
    num_days = state["num_days"]
    budget = state.get("budget", 0)
    preferences = state.get("user_preferences", {})
    pref_context = format_preferences_for_llm(preferences)
    system_message = SystemMessage(content=f"""You are a world-class travel researcher. 
Your task: Research {destination} for a {num_days}-day trip with a budget of ${budget:.2f}.
{pref_context}

Instructions:
1. First, think about what information you need to gather
2. Consider the user's preferences when formulating search queries
3. Generate 3-5 focused search queries to find the best travel information
4. Use the search_web tool to search for each query
5. Analyze all search results
6. Summarize the 10 most relevant and useful findings
7. Prioritize recommendations that match user preferences

You have access to the search_web tool. Use it multiple times to gather comprehensive information.
After completing your research, provide a clear summary of your findings.""")
    user_message = HumanMessage(
        content=f"Please research {destination} for a {num_days}-day trip with a budget of ${budget:.2f}. {pref_context}")
    llm = state["_llm"]
    llm_with_tools = llm.bind_tools([search_web])
    current_messages = [system_message, user_message]
    max_iterations = 10
    for iteration in range(max_iterations):
        response = llm_with_tools.invoke(current_messages)
        current_messages.append(response)
        if not response.tool_calls:
            break
        from langchain_core.messages import ToolMessage
        for tool_call in response.tool_calls:
            try:
                tool_result = search_web.invoke(tool_call["args"])
            except Exception as e:
                tool_result = f"Search failed: {str(e)}"
            tool_message = ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
            current_messages.append(tool_message)
    return {**state, "messages": [AIMessage(content=response.content, name="Researcher")]}


def planner_agent(state: AgentState) -> AgentState:
    destination = state["destination"]
    num_days = state["num_days"]
    budget = state.get("budget", 0)
    messages = state["messages"]
    preferences = state.get("user_preferences", {})
    research_results = ""
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.name == "Researcher":
            research_results = msg.content
            break
    pref_instructions = ""
    if preferences:
        pref_instructions = "\n\nIMPORTANT - CUSTOMIZE BASED ON USER PREFERENCES:"
        if preferences.get("travel_pace"):
            pace = preferences["travel_pace"]
            if "Relaxed" in pace:
                pref_instructions += "\n- Keep schedule relaxed with 2-3 activities per day and built-in rest time"
            elif "Packed" in pace:
                pref_instructions += "\n- Create a packed schedule with 6+ activities per day"
            else:
                pref_instructions += "\n- Maintain moderate pace with 4-5 activities per day"
        if preferences.get("traveling_with"):
            groups = preferences["traveling_with"]
            if "Family with young kids" in groups:
                pref_instructions += "\n- Prioritize kid-friendly activities, short travel distances, and frequent breaks"
            if "Elderly" in groups:
                pref_instructions += "\n- Choose easily accessible locations with minimal walking"
        if preferences.get("interests"):
            pref_instructions += f"\n- Focus heavily on: {', '.join(preferences['interests'])}"
        if preferences.get("dietary_restrictions"):
            pref_instructions += f"\n- Recommend restaurants that accommodate: {preferences['dietary_restrictions']}"
        if preferences.get("mobility"):
            mobility = preferences["mobility"]
            if "Wheelchair" in mobility or "less walking" in mobility:
                pref_instructions += "\n- Ensure all activities are wheelchair/mobility-friendly"
        if preferences.get("wake_up_time"):
            wake = preferences["wake_up_time"]
            if "Early bird" in wake:
                pref_instructions += "\n- Start activities early (7-8 AM) to beat crowds"
            elif "Late riser" in wake:
                pref_instructions += "\n- Schedule morning activities from 10-11 AM onwards"
        if preferences.get("additional_notes"):
            pref_instructions += f"\n- Special requests: {preferences['additional_notes']}"
    system_message = SystemMessage(content=f"""You are a senior travel planner.

Task: Create a detailed {num_days}-day itinerary for {destination} based on the research provided.
Budget: ${budget:.2f} total (${budget / num_days:.2f} per day)
{pref_instructions}

Requirements:
1. Structure the itinerary clearly by day (Day 1, Day 2, etc.)
2. Include specific activities, estimated times, and locations
3. Add practical tips (best times to visit, booking recommendations)
4. Balance popular attractions with local experiences
5. Consider travel time between locations
6. Include meal suggestions (considering dietary restrictions if any)
7. Keep activities within the daily budget of ${budget / num_days:.2f}
8. Mention estimated costs for major activities
9. **CRITICAL: Tailor the entire itinerary to match user preferences above**

Output ONLY a valid JSON object wrapped in ```json ... ``` code block. No other text before or after.

JSON schema:
{{
  "overview": "1 sentence trip summary",
  "days": [
    {{
      "day": 1,
      "title": "Day title, e.g. Ancient Rome & Baroque Grandeur",
      "city": "Rome",
      "activities": [
        {{
          "time": "9:00 AM",
          "slot": "Morning",
          "name": "Colosseum & Arch of Constantine",
          "description": "2-3 sentences about what to do, see, and experience at this location. Include practical details like which entrance to use or best photo spots.",
          "cost": "Free",
          "duration": "2h"
        }}
      ],
      "tips": ["Practical tip 1", "Practical tip 2"]
    }}
  ]
}}

Rules for JSON output:
- "slot" must be one of: "Morning", "Afternoon", "Evening"
- "cost" should be a short string: "Free", "€18", "€5-10", "$15" etc.
- "duration" should be like: "1h", "1.5h", "2h", "30min"
- "description" must be 2-3 complete sentences. Do NOT use sub-fields like "Activity:", "Location:", "Accessibility:". Merge all useful info into natural flowing sentences.
- Each day should have 3-5 activities covering morning, afternoon, and evening
- "tips" is an array of 1-3 short practical tips for that day
- Include meal recommendations as activities (Lunch, Dinner) with restaurant suggestions in the description
- **CRITICAL: Tailor the entire itinerary to match user preferences above**""")
    user_message = HumanMessage(
        content=f"Research Results:\n{research_results}\n\nPlease create a detailed itinerary based on this research and user preferences. Output ONLY the JSON.")
    llm = state["_llm"]
    response = llm.invoke([system_message, user_message])

    # 解析 JSON
    raw = response.content.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
    s, e = raw.find('{'), raw.rfind('}') + 1
    itinerary_json = None
    if s != -1 and e > s:
        try:
            itinerary_json = json.loads(raw[s:e])
        except json.JSONDecodeError:
            pass

    if itinerary_json and "days" in itinerary_json:
        # 生成纯文本版本供导出、地图、甘特图等使用
        text_version = ""
        for day in itinerary_json.get("days", []):
            text_version += f"\nDay {day['day']}: {day['title']} ({day.get('city','')})\n"
            for act in day.get("activities", []):
                text_version += f"- {act['slot']} ({act['time']}): {act['name']} - {act['description']} Cost: {act.get('cost','N/A')}. Duration: {act.get('duration','N/A')}.\n"
            if day.get("tips"):
                text_version += f"Tips: {'; '.join(day['tips'])}\n"
        final_itinerary = text_version
    else:
        itinerary_json = None
        final_itinerary = response.content

    return {**state, "messages": [AIMessage(content=response.content, name="Planner")],
            "final_itinerary": final_itinerary, "itinerary_json": itinerary_json}


def feedback_analyzer_agent(state: AgentState) -> AgentState:
    feedback = state.get("user_feedback", "")
    current_itinerary = state["final_itinerary"]
    if not feedback:
        return state
    system_message = SystemMessage(content=f"""You are a feedback analyzer. User has feedback on their itinerary:

CURRENT ITINERARY:
{current_itinerary}

USER FEEDBACK:
{feedback}

Analyze what specific changes are needed. Identify:
1. Which day(s) need modification (if specific days mentioned)
2. What kind of change (reduce activities, swap order, change type, add rest time, etc.)
3. What should remain unchanged

Output in JSON format:
{{
    "days_to_modify": [2, 3],
    "change_type": "reduce_activities",
    "specific_instruction": "Remove 2 activities from Day 2, keep morning temple visit."
}}

If no specific days mentioned, set "days_to_modify": "all".
""")
    llm = state["_llm"]
    response = llm.invoke([system_message, HumanMessage(content=feedback)])
    try:
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            analysis = json.loads(json_str)
        else:
            analysis = {"error": "Could not parse feedback", "raw_response": content}
    except Exception as e:
        analysis = {"error": str(e), "raw_response": response.content}
    return {**state, "feedback_analysis": analysis}


def revision_planner_agent(state: AgentState) -> AgentState:
    analysis = state.get("feedback_analysis", {})
    current_itinerary = state["final_itinerary"]
    destination = state["destination"]
    num_days = state["num_days"]
    budget = state["budget"]
    json_schema_instruction = """

Output ONLY a valid JSON object wrapped in ```json ... ``` code block. No other text.
Use the same JSON schema as the original:
{{"overview": "...", "days": [{{"day": 1, "title": "...", "city": "...", "activities": [{{"time": "9:00 AM", "slot": "Morning|Afternoon|Evening", "name": "...", "description": "2-3 sentences.", "cost": "Free|€18|...", "duration": "2h|..."}}], "tips": ["..."]}}]}}
"""

    if "error" in analysis:
        system_message = SystemMessage(content=f"""You are revising the itinerary based on user feedback.

USER FEEDBACK: {state.get('user_feedback', '')}

CURRENT ITINERARY:
{current_itinerary}

Please revise the itinerary according to the user's feedback while maintaining:
- The destination: {destination}
- Duration: {num_days} days
- Budget: ${budget:.2f}

Make the requested changes and keep everything else the same.
{json_schema_instruction}""")
    else:
        system_message = SystemMessage(content=f"""You are revising the itinerary based on user feedback.

ORIGINAL ITINERARY:
{current_itinerary}

REQUIRED CHANGES:
{json.dumps(analysis, indent=2)}

Instructions:
1. If days_to_modify is specific (like [2, 3]), ONLY modify those days
2. If days_to_modify is "all", revise the entire itinerary
3. Follow the specific_instruction precisely
4. Keep all other content EXACTLY the same
5. Keep within budget of ${budget:.2f}
{json_schema_instruction}""")
    llm = state["_llm"]
    revision_messages = [
        system_message,
        HumanMessage(content="Apply the instructions above and output ONLY the JSON."),
    ]
    response = llm.invoke(revision_messages)

    # 解析 JSON
    raw = response.content.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
    s, e = raw.find('{'), raw.rfind('}') + 1
    itinerary_json = None
    if s != -1 and e > s:
        try:
            itinerary_json = json.loads(raw[s:e])
        except json.JSONDecodeError:
            pass

    if itinerary_json and "days" in itinerary_json:
        text_version = ""
        for day in itinerary_json.get("days", []):
            text_version += f"\nDay {day['day']}: {day['title']} ({day.get('city','')})\n"
            for act in day.get("activities", []):
                text_version += f"- {act['slot']} ({act['time']}): {act['name']} - {act['description']} Cost: {act.get('cost','N/A')}. Duration: {act.get('duration','N/A')}.\n"
            if day.get("tips"):
                text_version += f"Tips: {'; '.join(day['tips'])}\n"
        final_itinerary = text_version
    else:
        itinerary_json = None
        final_itinerary = response.content

    return {
        **state,
        "final_itinerary": final_itinerary,
        "itinerary_json": itinerary_json,
        "iteration": state.get("iteration", 0) + 1,
        "user_feedback": ""
    }


# ==================== Restaurant Recommender Agent ====================

def extract_cities_from_itinerary(itinerary: str, main_destination: str, llm) -> list[str]:
    prompt = f"""From the travel itinerary below, extract every distinct CITY or AREA that is visited.
Return ONLY a JSON array of city/area name strings, e.g. ["Tokyo", "Kyoto", "Osaka"].
If the whole trip is in one city, return that single city.
Do not include neighbourhoods or individual attraction names — only cities or major areas.

ITINERARY:
{itinerary[:2000]}

Return only the JSON array, no other text."""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
        start, end = raw.find('['), raw.rfind(']') + 1
        if start != -1 and end > start:
            cities = json.loads(raw[start:end])
            if isinstance(cities, list) and cities:
                return [str(c) for c in cities]
    except Exception:
        pass
    return [main_destination]


def restaurant_recommender_agent(state: AgentState) -> AgentState:
    main_destination = state["destination"]
    num_days    = state["num_days"]
    itinerary   = state.get("final_itinerary", "")
    preferences = state.get("user_preferences", {})
    dietary     = preferences.get("dietary_restrictions", "")
    interests   = preferences.get("interests", [])
    budget      = state.get("budget", 0)
    daily_food  = (budget * 0.25) / max(num_days, 1)
    llm         = state["_llm"]

    cities = extract_cities_from_itinerary(itinerary, main_destination, llm)
    all_restaurants = []

    for city in cities:
        search_queries = [f"best restaurants {city} local food must try"]
        if dietary:
            search_queries.append(f"{dietary} friendly restaurants {city}")
        if "Food & Dining" in (interests or []):
            search_queries.append(f"hidden gem local favourite restaurants {city}")

        raw_results = []
        for q in search_queries:
            try:
                # raw_results.append(f"Query: {q}\n{serp_wrapper.run(q)}")  # SerpAPI
                result = tavily_client.search(q, max_results=3)
                text = "\n".join([r.get("content", "")[:300] for r in result.get("results", [])])
                raw_results.append(f"Query: {q}\n{text}")
            except Exception as e:
                raw_results.append(f"Query: {q}\nSearch failed: {e}")

        combined = "\n\n---\n\n".join(raw_results)

        dietary_note = f"Dietary restrictions to respect: {dietary}." if dietary else ""
        budget_note  = f"Daily food budget per person: approx ${daily_food:.0f}."

        user_prompt = f"""You are a restaurant curator for {city}.
{dietary_note}
{budget_note}

Based on the search results below, produce a JSON array of exactly 3 to 4 restaurant recommendations for {city}.
Cover: 1 Breakfast, 1 Lunch, 1-2 Dinner. Include a spread of price ranges.

Each item MUST use this exact schema:
{{
  "name": "Restaurant Name",
  "cuisine": "e.g. Japanese / Street Food",
  "meal_type": "Breakfast | Lunch | Dinner | Snack",
  "price_range": "$ | $$ | $$$ | $$$$",
  "description": "1-2 sentences on what makes it special",
  "tip": "One practical tip (max 15 words)",
  "city": "{city}"
}}

Return ONLY the raw JSON array. No markdown, no backticks, no extra text.

SEARCH RESULTS:
{combined[:3500]}
"""
        try:
            response = llm.invoke([
                SystemMessage(content="You are a restaurant curator. Return ONLY a raw JSON array."),
                HumanMessage(content=user_prompt),
            ])
            raw = response.content.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
            start, end = raw.find('['), raw.rfind(']') + 1
            if start != -1 and end > start:
                city_recs = json.loads(raw[start:end])
                for r in city_recs:
                    r.setdefault("city", city)
                all_restaurants.extend(city_recs)
        except Exception:
            for meal, price in [("Breakfast", "$"), ("Lunch", "$$"), ("Dinner", "$$$")]:
                all_restaurants.append({
                    "name": f"Local {meal} Spot", "cuisine": "Local cuisine", "meal_type": meal,
                    "price_range": price, "description": f"A well-regarded local spot for {meal.lower()} in {city}.",
                    "tip": "Ask your hotel concierge for the latest recommendation.", "city": city,
                })

    return {**state, "restaurant_recommendations": all_restaurants}


# ==================== City Guide Agent + PDF ====================

def city_guide_agent(state: AgentState) -> AgentState:
    itinerary  = state.get("final_itinerary", "")
    main_dest  = state["destination"]
    llm        = state["_llm"]

    cities = extract_cities_from_itinerary(itinerary, main_dest, llm)
    city_guides = {}

    for city in cities:
        queries = [
            f"{city} city overview history culture travel guide",
            f"{city} local food specialties traditions festivals",
            f"{city} travel tips etiquette currency transport tourist",
            f"{city} fun facts lesser known trivia interesting",
        ]
        raw_results = []
        for q in queries:
            try:
                # raw_results.append(f"[{q}]\n{serp_wrapper.run(q)}")  # SerpAPI
                result = tavily_client.search(q, max_results=3)
                text = "\n".join([r.get("content", "")[:300] for r in result.get("results", [])])
                raw_results.append(f"[{q}]\n{text}")
            except Exception as e:
                raw_results.append(f"[{q}]\nSearch failed: {e}")

        combined = "\n\n".join(raw_results)

        prompt = f"""You are writing a travel guidebook chapter about {city}.
Using the search results below, write a structured city guide in English.

Return ONLY a JSON object with these exact keys:
{{
  "overview":    "3-4 sentences: location, size, character, why visit",
  "history":     "3-4 sentences: key historical events and eras that shaped the city",
  "highlights":  ["5-7 bullet strings: top attractions, cuisine, culture, unique experiences"],
  "travel_tips": ["4-6 bullet strings: practical tips — currency, transport, etiquette, safety, best season"],
  "trivia":      ["4-5 bullet strings: surprising or lesser-known fun facts about the city"]
}}

No markdown, no backticks, no extra text — only the raw JSON object.

SEARCH RESULTS:
{combined[:5000]}
"""
        try:
            response = llm.invoke([
                SystemMessage(content="You are a travel guidebook writer. Return ONLY a raw JSON object."),
                HumanMessage(content=prompt),
            ])
            raw = response.content.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
            s, e = raw.find('{'), raw.rfind('}') + 1
            if s != -1 and e > s:
                guide = json.loads(raw[s:e])
                city_guides[city] = guide
                continue
        except Exception:
            pass

        # Fallback
        city_guides[city] = {
            "overview":    f"{city} is a fascinating destination worth exploring.",
            "history":     f"{city} has a rich history spanning many centuries.",
            "highlights":  [f"Explore the iconic landmarks of {city}", "Try the local cuisine", "Visit local markets"],
            "travel_tips": ["Check visa requirements before travel", "Use local currency for smaller purchases",
                            "Respect local customs and dress codes"],
            "trivia":      [f"{city} has many hidden gems waiting to be discovered",
                            "Local festivals offer unique cultural experiences"],
        }

    return {**state, "city_guides": city_guides}


def generate_travel_guide_pdf(destination, num_days, start_date, end_date,
                               city_guides, itinerary, budget_info, restaurants) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak,
        HRFlowable, Table, TableStyle,
    )
    from io import BytesIO

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2.2*cm, rightMargin=2.2*cm,
                            topMargin=2.5*cm, bottomMargin=2.5*cm,
                            title=f"Travel Guide: {destination}", author="AI Travel Planner")

    TEAL       = colors.HexColor("#4ECDC4")
    DARK_NAVY  = colors.HexColor("#1a1a2e")
    ORANGE     = colors.HexColor("#FF6B6B")
    MID_GREY   = colors.HexColor("#888888")

    base = getSampleStyleSheet()
    def S(name, **kw):
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    sty = {
        "cover_title": S("cover_title", fontSize=32, leading=38, textColor=colors.white,
                          fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=8),
        "ch_title": S("ch_title", fontSize=22, leading=28, textColor=DARK_NAVY,
                       fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=6),
        "section_hd": S("section_hd", fontSize=13, leading=18, textColor=TEAL,
                         fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4),
        "body": S("body", fontSize=10, leading=15, textColor=colors.HexColor("#333333"),
                   fontName="Helvetica", alignment=TA_JUSTIFY, spaceAfter=6),
        "bullet": S("bullet", fontSize=10, leading=14, textColor=colors.HexColor("#333333"),
                     fontName="Helvetica", leftIndent=12, spaceAfter=3),
        "toc_entry": S("toc_entry", fontSize=11, leading=18, textColor=DARK_NAVY,
                        fontName="Helvetica", spaceAfter=2),
        "footer_note": S("footer_note", fontSize=8, leading=11, textColor=MID_GREY,
                          fontName="Helvetica-Oblique", alignment=TA_CENTER),
        "rest_name": S("rest_name", fontSize=10, leading=13, textColor=DARK_NAVY, fontName="Helvetica-Bold"),
        "rest_meta": S("rest_meta", fontSize=9, leading=12, textColor=MID_GREY, fontName="Helvetica-Oblique"),
        "rest_desc": S("rest_desc", fontSize=9, leading=13, textColor=colors.HexColor("#444444"), fontName="Helvetica"),
    }

    story = []
    W = A4[0] - 4.4*cm

    def hr(color=TEAL, thickness=1.2):
        return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=8, spaceBefore=4)

    def bullets(items, icon="•"):
        elems = []
        for item in items:
            clean = re.sub(r'^[-•*]\s*', '', str(item))
            elems.append(Paragraph(f"{icon}  {clean}", sty["bullet"]))
        return elems

    # Cover
    cover_table = Table([[Paragraph(f"✈  {destination}", sty["cover_title"])]], colWidths=[W])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), DARK_NAVY), ("ROUNDEDCORNERS", [12]),
        ("TOPPADDING", (0,0), (-1,-1), 50), ("BOTTOMPADDING", (0,0), (-1,-1), 50),
        ("LEFTPADDING", (0,0), (-1,-1), 20), ("RIGHTPADDING", (0,0), (-1,-1), 20),
    ]))
    story.append(Spacer(1, 1.5*cm))
    story.append(cover_table)
    story.append(Spacer(1, 0.6*cm))

    date_str = ""
    if start_date and end_date:
        try:
            date_str = f"{start_date.strftime('%B %d')} – {end_date.strftime('%B %d, %Y')}"
        except Exception:
            date_str = ""
    sub_parts = [f"{num_days}-Day Journey"]
    if date_str:
        sub_parts.append(date_str)
    story.append(Paragraph(" · ".join(sub_parts), S("cs2", fontSize=13, leading=18, textColor=DARK_NAVY,
                                                      fontName="Helvetica", alignment=TA_CENTER, spaceAfter=4)))
    story.append(Paragraph(f"Generated by AI Travel Planner  ·  {datetime.now().strftime('%Y-%m-%d')}", sty["footer_note"]))
    story.append(Spacer(1, 0.5*cm))
    story.append(hr(ORANGE, 2))
    story.append(Spacer(1, 0.3*cm))

    city_list = ", ".join(city_guides.keys()) if city_guides else destination
    story.append(Paragraph(
        f"This guide covers {city_list} — including city overviews, history, "
        f"local highlights, practical travel tips, and fun trivia to enrich your journey.", sty["body"]))
    story.append(PageBreak())

    # Table of Contents
    story.append(Paragraph("Contents", sty["ch_title"]))
    story.append(hr())
    for i, city in enumerate(city_guides.keys(), 1):
        story.append(Paragraph(f"{i}.  {city}", sty["toc_entry"]))
    if restaurants:
        story.append(Paragraph(f"{len(city_guides)+1}.  Restaurant Recommendations", sty["toc_entry"]))
    story.append(PageBreak())

    # City chapters
    for city, guide in city_guides.items():
        ch_table = Table([[Paragraph(f"📍  {city}", S("ch_hdr", fontSize=20, leading=26, textColor=colors.white,
                                                        fontName="Helvetica-Bold"))]], colWidths=[W])
        ch_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), TEAL),
            ("TOPPADDING", (0,0), (-1,-1), 14), ("BOTTOMPADDING", (0,0), (-1,-1), 14),
            ("LEFTPADDING", (0,0), (-1,-1), 16),
        ]))
        story.append(ch_table)
        story.append(Spacer(1, 0.3*cm))

        story.append(Paragraph("City Overview", sty["section_hd"]))
        story.append(hr(TEAL, 0.8))
        story.append(Paragraph(str(guide.get("overview", "")), sty["body"]))

        story.append(Paragraph("Historical Background", sty["section_hd"]))
        story.append(hr(TEAL, 0.8))
        story.append(Paragraph(str(guide.get("history", "")), sty["body"]))

        for section_key, section_title, bullet_icon in [
            ("highlights", "Local Highlights", "★"),
            ("travel_tips", "Travel Tips", "✓"),
            ("trivia", "Fun Facts & Trivia", "💡"),
        ]:
            items = guide.get(section_key, [])
            if items:
                story.append(Paragraph(section_title, sty["section_hd"]))
                story.append(hr(TEAL, 0.8))
                story.extend(bullets(items, bullet_icon))

        story.append(Spacer(1, 0.4*cm))
        story.append(PageBreak())

    # Restaurant Recommendations in PDF
    if restaurants:
        story.append(Paragraph("Restaurant Recommendations", sty["ch_title"]))
        story.append(hr(ORANGE, 1.5))
        story.append(Paragraph("Curated dining picks for each city.", sty["body"]))
        story.append(Spacer(1, 0.3*cm))

        rest_by_city: dict[str, dict[str, list]] = {}
        meal_order_pdf = ["Breakfast", "Brunch", "Lunch", "Snack", "Dinner", "Any"]
        for r in restaurants:
            c = r.get("city") or r.get("destination", "Other")
            m = r.get("meal_type", "Any")
            rest_by_city.setdefault(c, {}).setdefault(m, []).append(r)

        for c, meal_dict in rest_by_city.items():
            story.append(Paragraph(f"📍  {c}", S("city_rest_hd", fontSize=14, leading=20, textColor=DARK_NAVY,
                                                   fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)))
            story.append(hr(TEAL, 0.6))

            for meal in meal_order_pdf:
                if meal not in meal_dict:
                    continue
                meal_icon_map = {"Breakfast":"🌅","Lunch":"☀️","Dinner":"🌙","Brunch":"🍳","Snack":"🍡","Any":"🍽️"}
                story.append(Paragraph(f"{meal_icon_map.get(meal,'🍽️')}  {meal}",
                    S("meal_hd", fontSize=11, leading=16, textColor=TEAL,
                      fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3)))

                for r in meal_dict[meal]:
                    meta = f"{r.get('price_range','$$')}  ·  {r.get('cuisine','')}"
                    story.append(Paragraph(r["name"], sty["rest_name"]))
                    story.append(Paragraph(meta, sty["rest_meta"]))
                    if r.get("description"):
                        story.append(Paragraph(r["description"], sty["rest_desc"]))
                    if r.get("tip"):
                        story.append(Paragraph(f"💡 {r['tip']}", sty["rest_desc"]))
                    story.append(Spacer(1, 0.15*cm))

        story.append(PageBreak())

    # Back page
    story.append(Spacer(1, 6*cm))
    story.append(hr(ORANGE, 2))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Have a wonderful trip! ✈️", S("back", fontSize=18, leading=24, textColor=DARK_NAVY,
                                                            fontName="Helvetica-Bold", alignment=TA_CENTER)))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Generated by AI Travel Planner · Powered by LangGraph + Google Gemini", sty["footer_note"]))

    doc.build(story)
    return buf.getvalue()


# ==================== Build Graph ====================
def route_entry(state: AgentState) -> str:
    """Branch: revise existing itinerary vs full new-trip pipeline."""
    fb = (state.get("user_feedback") or "").strip()
    it = (state.get("final_itinerary") or "").strip()
    if fb and it:
        return "revise"
    return "new"


def create_travel_agent_graph(gemini_api_key: str, serp_api_key: str = ""):
    # global serp_wrapper
    # serp_wrapper = SerpAPIWrapper(serpapi_api_key=serp_api_key)
    global tavily_client
    tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))
    llm = ChatGoogleGenerativeAI(model=MAIN_MODEL, google_api_key=gemini_api_key, temperature=0.7)
    workflow = StateGraph(AgentState)

    def wrap(fn):
        def inner(state):
            state["_llm"] = llm
            return fn(state)
        return inner

    def entry_pass(state: AgentState) -> dict:
        return {}

    workflow.add_node("entry", entry_pass)
    workflow.add_node("researcher", wrap(researcher_agent))
    workflow.add_node("planner", wrap(planner_agent))
    workflow.add_node("feedback_analyzer", wrap(feedback_analyzer_agent))
    workflow.add_node("revision_planner", wrap(revision_planner_agent))
    workflow.add_node("restaurant_recommender", wrap(restaurant_recommender_agent))
    workflow.add_node("city_guide", wrap(city_guide_agent))

    workflow.set_entry_point("entry")
    workflow.add_conditional_edges(
        "entry",
        route_entry,
        {"new": "researcher", "revise": "feedback_analyzer"},
    )
    workflow.add_edge("researcher", "planner")
    workflow.add_edge("planner", "restaurant_recommender")
    workflow.add_edge("restaurant_recommender", "city_guide")
    workflow.add_edge("city_guide", END)
    workflow.add_edge("feedback_analyzer", "revision_planner")
    workflow.add_edge("revision_planner", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ==================== Streamlit UI ====================
st.set_page_config(page_title="闺蜜旅行搭子", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* ── 全局色彩变量（mockup 配色） ──────────────── */
    :root {
        --primary:     #7C3AED;
        --primary-light:#EDE9FE;
        --primary-bg:  #F5F3FF;
        --pink:        #7C3AED;
        --pink-light:  #EDE9FE;
        --pink-bg:     #F5F3FF;
        --lavender:    #8B5CF6;
        --lavender-bg: #F5F3FF;
        --accent-pink: #EC4899;
        --accent-amber:#D97706;
        --mint:        #34D399;
        --mint-bg:     #ECFDF5;
        --peach:       #FBBF24;
        --sky:         #60A5FA;
        --navy:        #1F2937;
        --grey-50:     #FAFAFA;
        --grey-100:    #F3F4F6;
        --grey-200:    #E5E7EB;
        --grey-400:    #9CA3AF;
        --grey-500:    #6B7280;
        --grey-700:    #374151;
        --radius-sm:   8px;
        --radius-md:   10px;
        --radius-lg:   16px;
        --shadow-sm:   0 1px 2px rgba(0,0,0,0.06);
        --shadow-md:   0 4px 12px rgba(0,0,0,0.08);
        --shadow-hover:0 6px 20px rgba(124,58,237,0.12);
    }

    /* ── 全局间距压缩（一屏内完成） ─────────────── */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0 !important;
        max-width: 1200px !important;
    }
    [data-testid="stVerticalBlock"] > div:has(> [data-testid="stVerticalBlockBorderWrapper"]) {
        gap: 0.35rem !important;
    }
    .stMarkdown { margin-bottom: 0 !important; }
    .element-container { margin-bottom: 0.2rem !important; }

    /* ── 排版 ─────────────────────────────────────── */
    h1 {
        text-align: center;
        margin-bottom: 0.1rem;
        color: var(--navy);
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h3 {
        color: var(--navy);
        font-weight: 700;
        letter-spacing: -0.3px;
        margin-top: 0.3rem !important;
        margin-bottom: 0.2rem !important;
    }
    .stCaption {
        text-align: center;
        font-size: 0.92rem;
        color: var(--grey-500);
        margin-bottom: 0.5rem;
    }

    /* ── 主内容区 ──────────────────────────────────── */
    .itinerary-content {
        background: var(--grey-50);
        padding: 2rem;
        border-radius: var(--radius-md);
        margin-top: 1rem;
        border: 1px solid var(--grey-200);
    }
    .budget-card {
        background: linear-gradient(135deg, var(--pink) 0%, var(--lavender) 100%);
        color: white;
        padding: 1.8rem;
        border-radius: var(--radius-md);
        margin: 1rem 0;
        box-shadow: var(--shadow-md);
    }

    /* ── 按钮 ──────────────────────────────────────── */
    .stButton>button {
        width: 100%;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-hover) !important;
    }
    .stButton>button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #8B5CF6, #EC4899) !important;
        border: none !important;
        color: white !important;
    }

    /* ── 侧边栏 ───────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--pink-bg) 0%, #FFFFFF 40%) !important;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--navy) !important;
    }

    /* ── 输入控件 ──────────────────────────────────── */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div {
        border-radius: var(--radius-sm) !important;
    }
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: var(--pink) !important;
        box-shadow: 0 0 0 2px var(--pink-light) !important;
    }

    /* ── Expander ──────────────────────────────────── */
    .streamlit-expanderHeader {
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
    }

    /* 行程卡片里的景点照片 */
    [data-testid="stExpander"] img {
        border-radius: 8px;
        object-fit: cover;
        max-height: 100px;
    }

    /* ── Metric 卡片 ──────────────────────────────── */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--grey-200);
        border-radius: var(--radius-md);
        padding: 14px 18px;
        box-shadow: var(--shadow-sm);
    }
    [data-testid="stMetricLabel"] {
        color: var(--grey-500) !important;
        font-size: 0.82rem !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--navy) !important;
        font-weight: 700 !important;
    }

    /* ── Tab 美化 ──────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--pink-light) !important;
        color: var(--pink) !important;
    }

    /* ── 分割线 ─────────────────────────────────────── */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--grey-200), transparent);
        margin: 1.5rem 0;
    }

    /* ── 模板卡片 ──────────────────────────────────── */
    .template-card {
        background: white;
        border: 1.5px solid var(--grey-200);
        border-radius: var(--radius-md);
        padding: 20px 16px;
        text-align: center;
        transition: all 0.25s ease;
        cursor: pointer;
        height: 100%;
    }
    .template-card:hover {
        border-color: var(--pink);
        box-shadow: var(--shadow-hover);
        transform: translateY(-3px);
    }
    .template-card .icon {
        font-size: 2rem;
        margin-bottom: 8px;
    }
    .template-card .title {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--navy);
        margin-bottom: 4px;
    }
    .template-card .desc {
        font-size: 0.78rem;
        color: var(--grey-500);
        line-height: 1.4;
    }
    .template-card .tags {
        margin-top: 8px;
    }
    .template-card .tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.68rem;
        font-weight: 600;
        margin: 2px;
        background: var(--pink-light);
        color: var(--pink);
    }

    /* ── 确保 block-container 不裁切导航栏 ──────────── */
    .block-container { overflow: visible !important; }

    /* ── 顶部导航栏 ─────────────────────────────────── */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid var(--grey-200);
        margin: 0 0 0.6rem;
        position: relative;
        z-index: 10;
        white-space: nowrap;
        flex-wrap: nowrap;
        gap: 16px;
    }
    .top-nav .logo {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-shrink: 0;
    }
    .top-nav .logo-icon {
        width: 28px; height: 28px;
        border-radius: 8px;
        background: linear-gradient(135deg, #8B5CF6, #EC4899);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .top-nav .logo-text {
        font-size: 15px;
        font-weight: 500;
        color: var(--navy);
    }
    /* seg-ctrl 已移至 st.radio，这里留空 */
    .top-nav .step-ind {
        display: flex;
        align-items: center;
        gap: 6px;
        flex-shrink: 0;
    }
    .top-nav .step-dot {
        width: 20px; height: 20px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 10px;
        font-weight: 500;
    }
    .top-nav .step-dot.active {
        background: var(--primary);
        color: #fff;
    }
    .top-nav .step-dot.pending {
        border: 1.5px solid #D1D5DB;
        color: var(--grey-400);
        background: transparent;
    }
    .top-nav .step-dot.done {
        background: var(--primary);
        color: #fff;
    }
    .top-nav .step-label {
        font-size: 12px;
        font-weight: 500;
    }
    .top-nav .step-label.active { color: var(--primary); }
    .top-nav .step-label.pending { color: var(--grey-400); }
    .top-nav .step-label.done { color: var(--primary); }
    .top-nav .step-line {
        width: 20px;
        height: 1px;
        background: #D1D5DB;
    }

    /* ── 历史卡片 ──────────────────────────────────── */
    .history-item {
        background: #ffffff;
        border: 1px solid var(--grey-200);
        padding: 1rem;
        border-radius: var(--radius-md);
        margin-bottom: 0.5rem;
        transition: box-shadow 0.2s ease;
    }
    .history-item:hover {
        box-shadow: var(--shadow-md);
    }

    /* ── 右侧视觉区 ──────────────────────────────── */
    .visual-panel {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(160deg, #F5F3FF 0%, #FDF2F8 50%, #F0F9FF 100%);
        border-radius: var(--radius-lg);
        padding: 32px;
        min-height: 340px;
    }
    .visual-panel .dest-name {
        font-size: 20px;
        font-weight: 500;
        color: #4C1D95;
        margin: 16px 0 4px;
        text-align: center;
    }
    .visual-panel .dest-sub {
        font-size: 13px;
        color: var(--primary);
        margin: 0 0 14px;
        text-align: center;
    }
    .visual-panel .tag-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        justify-content: center;
    }
    .visual-panel .tag {
        font-size: 11px;
        padding: 4px 12px;
        border-radius: 20px;
    }

    /* ── 总览卡片 (Step 3) ────────────────────────── */
    .summary-card {
        background: white;
        border: 1.5px solid var(--grey-200);
        border-radius: var(--radius-md);
        padding: 18px 24px;
        box-shadow: var(--shadow-sm);
        margin: 0.3rem 0;
    }
    .summary-card .row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
        font-size: 0.9rem;
        color: var(--grey-700);
    }
    .summary-card .row .label {
        color: var(--grey-500);
        min-width: 70px;
        font-size: 0.82rem;
    }
    .summary-card .row .value {
        font-weight: 600;
        color: var(--navy);
    }
    .summary-card .pref-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 4px;
    }
    .summary-card .pref-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        background: var(--primary-light);
        color: var(--primary);
    }

    /* (mode-switch-radio 已删除，改用 st.button) */
</style>
""", unsafe_allow_html=True)

# Initialize session state
for key, default in [
    ('itinerary', None), ('budget_info', None), ('history', []),
    ('current_destination', ""), ('current_days', 7),
    ('start_date', datetime.today()), ('end_date', datetime.today() + timedelta(days=7)),
    ('template_applied', False), ('packing_list', None),
    ('current_thread_id', datetime.now().strftime("%Y%m%d_%H%M%S")),
    ('weather_data', None), ('user_preferences', {}),
    ('restaurant_recommendations', []), ('city_guides', {}),
    ('photo_recognition_results', []),
    ('fact_check_results', []),
    ('itinerary_json', None),
    ('wizard_step', 1),
    ('wizard_budget_level', 'medium'),
    ('wizard_currency', 'USD'),
    ('wizard_custom_budget', None),
    ('home_mode', 'plan'),          # "plan" | "vibe"
    ('vibe_messages', []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:12px;">
        <span style="font-size:1.5rem;">🌸</span>
        <div style="font-size:0.85rem; font-weight:600; color:#1E293B; margin-top:4px;">闺蜜旅行搭子</div>
        <div style="font-size:0.72rem; color:#94A3B8;">Powered by AI</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    # 优先从 .env / 环境变量读取，侧边栏输入可覆盖
    _env_gemini  = os.environ.get("GEMINI_API_KEY", "")
    # _env_serp    = os.environ.get("SERPAPI_API_KEY", "")  # SerpAPI 暂停
    _env_tavily  = os.environ.get("TAVILY_API_KEY", "")
    _env_stepfun = os.environ.get("STEPFUN_API_KEY", "")
    _env_weather = os.environ.get("OPENWEATHER_API_KEY", "")

    _has_env = any([_env_gemini, _env_tavily])
    if _has_env:
        st.markdown("##### ✅ API 已从 .env 加载")
        st.caption("如需覆盖，展开下方手动输入")
        with st.expander("🔑 手动输入 API Key", expanded=False):
            _gemini_input  = st.text_input("Google Gemini API Key", type="password", value="",
                                           help="留空则使用 .env 中的值")
            # _serp_input    = st.text_input("SerpAPI Key", type="password", value="",
            #                                help="留空则使用 .env 中的值")
            _tavily_input  = st.text_input("Tavily API Key", type="password", value="",
                                           help="留空则使用 .env 中的值")
            _stepfun_input = st.text_input("StepFun API Key (语音合成)", type="password", value="",
                                           help="留空则使用 .env 中的值")
            _weather_input = st.text_input("OpenWeatherMap API Key", type="password", value="",
                                           help="留空则使用 .env 中的值")
    else:
        st.markdown("##### 🔑 API 配置")
        st.caption("填一次后可存到 .env 文件，下次自动加载")
        _gemini_input  = st.text_input("Google Gemini API Key", type="password",
                                       help="从 https://makersuite.google.com/app/apikey 获取")
        # _serp_input    = st.text_input("SerpAPI Key", type="password",
        #                                help="从 https://serpapi.com 获取")
        _tavily_input  = st.text_input("Tavily API Key", type="password",
                                       help="从 https://tavily.com 获取")
        _stepfun_input = st.text_input("StepFun API Key (语音合成)", type="password",
                                       help="从 https://platform.stepfun.com 获取")
        with st.expander("🌤️ Weather API（可选）"):
            _weather_input = st.text_input("OpenWeatherMap API Key", type="password",
                                           help="从 https://openweathermap.org/api 获取免费 API")
            st.caption("留空则使用模拟天气数据")

    # 合并：手动输入优先，否则用环境变量
    gemini_api_key  = _gemini_input.strip()  or _env_gemini
    # serp_api_key    = _serp_input.strip()    or _env_serp  # SerpAPI 暂停
    serp_api_key    = ""  # 保留变量名兼容
    tavily_api_key  = _tavily_input.strip()  or _env_tavily
    stepfun_api_key = _stepfun_input.strip() or _env_stepfun
    weather_api_key = _weather_input.strip() or _env_weather

    st.markdown("---")

    if gemini_api_key:
        initialize_chat_assistant(gemini_api_key)
        display_chat_assistant()
        st.markdown("---")

    st.markdown("### 📖 怎么用？")
    st.markdown("""
    1. 🔑 输入 API Keys
    2. ⚡ 选模板 或 自己填
    3. ✨ 填写偏好（可选）
    4. 🚀 一键生成攻略
    5. 📅 查看时间线 + 🗺️ 地图
    6. 📸 传照片 → AI识别景点
    7. 🔍 粘贴小红书 → 辟谣校验
    8. 🍽️ 餐厅推荐 + 📖 下载PDF
    9. 🔄 不满意？直接改！
    """)

    st.markdown("---")
    st.markdown("### 🤖 AI 团队")
    st.markdown("""
    <div style="font-size: 0.85rem; line-height: 1.8;">
    🔍 <b>研究员</b> — 全网搜集目的地资讯<br>
    📝 <b>规划师</b> — 定制你的专属行程<br>
    🍽️ <b>美食顾问</b> — 挖掘当地宝藏餐厅<br>
    📖 <b>导游</b> — 城市历史文化科普<br>
    📸 <b>摄影助手</b> — 识别照片中的景点<br>
    🔍 <b>辟谣官</b> — 小红书种草帖校验<br>
    💬 <b>小助手</b> — 随时答疑解惑
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.checkbox("📊 Compare Trips", key="show_comparison"):
        display_itinerary_comparison()

    st.markdown("---")
    display_statistics_dashboard()

    st.markdown("---")
    st.markdown("### 📚 History")
    if st.session_state.history:
        for idx, item in enumerate(st.session_state.history):
            title = f"🗓️ {item['destination']} - {item['num_days']}days"
            if item.get('start_date'):
                title += f" ({item['start_date']})"
            with st.expander(title, expanded=False):
                st.caption(f"created time: {item['timestamp']}")
                st.caption(f"budget: ${item['budget']['total']:.0f}")
                if item.get('start_date') and item.get('end_date'):
                    st.caption(f"date of travel: {item['start_date']} ~ {item['end_date']}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("load", key=f"load_{item['id']}"):
                        load_history_item(item['id'])
                        st.rerun()
                with col2:
                    if st.button("delete", key=f"del_{item['id']}"):
                        delete_history_item(item['id'])
                        st.rerun()
    else:
        st.info("暂无历史记录")

# ==================== Stepper Helper ====================
def render_nav_bar(current_step: int):
    """渲染顶部导航栏：logo（左）+ 步骤指示器（右）。"""
    step_labels = ["行程", "偏好", "确认"]
    steps_html = ""
    for i, lbl in enumerate(step_labels):
        num = i + 1
        if num < current_step:
            dot_cls, lbl_cls = "done", "done"
        elif num == current_step:
            dot_cls, lbl_cls = "active", "active"
        else:
            dot_cls, lbl_cls = "pending", "pending"
        steps_html += (
            f'<div style="display:flex;align-items:center;gap:4px;">'
            f'<div class="step-dot {dot_cls}">{num}</div>'
            f'<span class="step-label {lbl_cls}">{lbl}</span></div>'
        )
        if i < len(step_labels) - 1:
            steps_html += '<div class="step-line"></div>'

    logo_svg = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>'

    nav_html = (
        '<div class="top-nav">'
        '<div class="logo">'
        f'<div class="logo-icon">{logo_svg}</div>'
        '<span class="logo-text">闺蜜旅行搭子</span>'
        '</div>'
        f'<div class="step-ind">{steps_html}</div>'
        '</div>'
    )
    st.markdown(nav_html, unsafe_allow_html=True)


# ==================== Result Page ====================
def parse_itinerary_to_days(itinerary: str, start_date: datetime, num_days: int) -> list[dict]:
    """把 Gemini 输出的 markdown 行程解析成结构化的 day 列表。"""
    # 去掉第一个 Day 之前的废话
    first_day = re.search(r'(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*{1,2})?(?:Day\s*\d+|第\s*\d+\s*天)', itinerary, re.IGNORECASE)
    if first_day:
        itinerary = itinerary[first_day.start():]

    # 按 Day N / 第N天 拆分
    day_pattern = re.compile(
        r'(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*{1,2})?(?:Day\s*(\d+)|第\s*(\d+)\s*天)\s*[:\-–—]?\s*(?:\*{1,2})?\s*([^\n]*)\n(.*?)(?=\n\s*(?:#{1,3}\s*)?(?:\*{1,2})?(?:Day\s*\d+|第\s*\d+\s*天)|\Z)',
        re.DOTALL | re.IGNORECASE
    )

    def extract_cost_tags(text: str) -> list[str]:
        tags = []
        for p in re.findall(r'(?:Free|免费|€[\d,]+(?:\s*-\s*€[\d,]+)?|£[\d,]+|\$[\d,]+|¥[\d,]+)', text, re.IGNORECASE):
            tags.append(p.strip())
        return tags

    def truncate_desc(text: str, limit: int = 120) -> str:
        if len(text) <= limit:
            return text
        # 在最后一个完整句号处截断
        truncated = text[:limit]
        last_period = max(truncated.rfind('.'), truncated.rfind('。'), truncated.rfind(','))
        if last_period > limit * 0.5:
            return truncated[:last_period + 1] + "..."
        # 在最后一个空格处截断（不截断单词）
        last_space = truncated.rfind(' ')
        if last_space > limit * 0.5:
            return truncated[:last_space] + "..."
        return truncated + "..."

    # 顶级活动行：-   **Morning (9:00 AM): Activity Title**
    activity_header_re = re.compile(
        r'^-\s+\*{0,2}(?:Morning|Lunch|Afternoon|Evening|Night|Dinner|Breakfast|Mid-Morning|Late Morning|Late Afternoon)\s*'
        r'(?:\(([^)]*)\))?\s*[:：]\s*(.+?)(?:\*{0,2})\s*$',
        re.IGNORECASE
    )

    # 子字段行：    *   **Activity:** ... 或 *   **Estimated Cost:** ...
    sub_field_re = re.compile(
        r'^\s*[*•]\s*\**(?:Activity|Location|Estimated\s*Cost|Cost|Price|Accessibility|Duration|Booking|Address|Transport|Getting\s*There|Tip)\**\s*[:：]\s*(.*)',
        re.IGNORECASE
    )

    # Tips 行
    tips_re = re.compile(
        r'^\s*[*•-]\s*\**(?:Tips?\s*(?:and\s*)?Notes?|Notes?\s*(?:and\s*)?Tips?|Tip|注意)\**\s*[:：]\s*(.*)',
        re.IGNORECASE
    )

    slot_map = {
        "morning": "Morning", "breakfast": "Morning", "mid-morning": "Morning", "late morning": "Morning",
        "lunch": "Afternoon", "afternoon": "Afternoon", "midday": "Afternoon", "late afternoon": "Afternoon",
        "evening": "Evening", "dinner": "Evening", "night": "Evening",
    }

    def detect_slot(text: str) -> str:
        t = text.lower()
        for kw, slot in slot_map.items():
            if kw in t:
                return slot
        return ""

    days = []
    for m in day_pattern.finditer(itinerary):
        day_num = int(m.group(1) or m.group(2))
        title = re.sub(r'\*+', '', m.group(3)).strip(" -:：–—")
        content = m.group(4)

        date_obj = start_date + timedelta(days=day_num - 1)
        date_str = date_obj.strftime("%A, %b %d")

        # 提取 Tips
        tips = []
        for tip_m in tips_re.finditer(content):
            tip_text = re.sub(r'\*+', '', tip_m.group(1)).strip()
            if tip_text:
                tips.append(tip_text)

        # 按活动 header 拆分
        slots = {"Morning": [], "Afternoon": [], "Evening": []}
        current_slot = "Morning"
        current_activity = None

        def flush_activity():
            nonlocal current_activity
            if current_activity and current_activity.get("name"):
                current_activity["desc"] = truncate_desc(current_activity["desc"])
                slots[current_slot].append(current_activity)
            current_activity = None

        for line in content.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # 跳过 *Focus: ...* 行
            if re.match(r'^\*Focus\s*:', line_stripped, re.IGNORECASE):
                continue
            # 跳过 tips 行（已单独提取）
            if tips_re.match(line_stripped):
                continue
            # 跳过纯标题行
            if re.match(r'^#{1,3}\s', line_stripped):
                continue

            # 1) 顶级活动 header：-   **Morning (9:00 AM): Title**
            header_match = activity_header_re.match(line_stripped)
            if header_match:
                flush_activity()
                detected = detect_slot(line_stripped)
                if detected:
                    current_slot = detected
                time_str = header_match.group(1) or ""
                act_name = re.sub(r'\*+', '', header_match.group(2)).strip(" -:：–—")
                current_activity = {
                    "time": time_str.strip(),
                    "name": act_name[:60],
                    "desc": "",
                    "tags": [],
                }
                continue

            # 2) 子字段行
            sub_match = sub_field_re.match(line_stripped)
            if sub_match and current_activity is not None:
                value = re.sub(r'\*+', '', sub_match.group(1)).strip()
                field_lower = line_stripped.lower()
                if 'activity' in field_lower:
                    if current_activity["desc"]:
                        current_activity["desc"] += " " + value
                    else:
                        current_activity["desc"] = value
                elif 'cost' in field_lower or 'price' in field_lower:
                    current_activity["tags"].extend(extract_cost_tags(value))
                    if not current_activity["tags"] and value:
                        current_activity["tags"].append(value[:20])
                # Location / Accessibility / Duration 等忽略
                continue

            # 3) 非子字段的缩进 * 行（在 Activity: 之前的描述文字）
            if re.match(r'^\s*[*•]\s+', line_stripped) and current_activity is not None:
                # 如果当前活动还没有 desc，这可能是活动描述
                if not sub_field_re.match(line_stripped):
                    text = re.sub(r'^\s*[*•]\s+', '', line_stripped)
                    text = re.sub(r'\*+', '', text).strip()
                    if text and len(text) > 5 and not current_activity["desc"]:
                        current_activity["desc"] = text
                continue

        # flush 最后一个活动
        flush_activity()

        # fallback：如果没有解析到任何活动，按行均分
        if not any(slots.values()):
            all_acts = []
            for line in content.splitlines():
                ls = line.strip()
                if not ls or len(ls) < 6:
                    continue
                if re.match(r'^#{1,3}\s', ls) or tips_re.match(ls):
                    continue
                if re.match(r'^[-•*]\s', ls) or re.match(r'^\d+[.)]\s', ls):
                    clean = re.sub(r'^[-•*\d:.()\s]+', '', ls)
                    clean = re.sub(r'\*+', '', clean).strip()
                    if clean and len(clean) >= 4:
                        all_acts.append({"time": "", "name": clean[:60], "desc": "", "tags": extract_cost_tags(clean)})
            if all_acts:
                n = len(all_acts)
                t1, t2 = max(1, n // 3), max(2, n * 2 // 3)
                slots["Morning"] = all_acts[:t1]
                slots["Afternoon"] = all_acts[t1:t2]
                slots["Evening"] = all_acts[t2:]

        activity_count = sum(len(v) for v in slots.values())
        days.append({
            "day": day_num,
            "title": title if title else f"Day {day_num}",
            "date_str": date_str,
            "slots": slots,
            "tips": tips,
            "activity_count": activity_count,
        })

    return days


def render_day_cards(days: list[dict]):
    """把 parse 结果渲染成可折叠的 Day 卡片。"""
    slot_colors = {
        "Morning": ("#FEF3C7", "#92400E", "☀️"),
        "Afternoon": ("#D1FAE5", "#065F46", "🌤️"),
        "Evening": ("#EDE9FE", "#5B21B6", "🌙"),
    }

    for day_data in days:
        day_num = day_data["day"]
        color = DAY_COLORS[(day_num - 1) % len(DAY_COLORS)]
        badge = f'<span style="display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.7rem;font-weight:600;background:{color}22;color:{color};margin-left:8px;">{day_data["activity_count"]} 项活动</span>'

        header = f"Day {day_num} · {day_data['date_str']} — {day_data['title']}"
        with st.expander(header, expanded=(day_num == 1)):
            for slot_name in ["Morning", "Afternoon", "Evening"]:
                activities = day_data["slots"].get(slot_name, [])
                if not activities:
                    continue
                bg, fg, icon = slot_colors[slot_name]
                slot_label = {"Morning": "上午", "Afternoon": "下午", "Evening": "晚上"}[slot_name]

                # 时段标签
                st.markdown(
                    f'<div style="display:inline-block;padding:2px 10px;border-radius:6px;'
                    f'font-size:0.75rem;font-weight:600;background:{bg};color:{fg};margin:6px 0 4px;">'
                    f'{icon} {slot_label}</div>',
                    unsafe_allow_html=True)

                # 活动列表
                for act in activities:
                    time_html = f'<span style="color:#9CA3AF;font-size:0.8rem;min-width:60px;display:inline-block;">{act["time"]}</span> ' if act["time"] else ''
                    name_html = f'<span style="font-weight:600;color:#1F2937;font-size:0.88rem;">{act["name"]}</span>'
                    tags_html = ""
                    for tag in act.get("tags", []):
                        tag_color = "#059669" if "free" in tag.lower() or "免费" in tag else "#7C3AED"
                        tags_html += f'<span style="display:inline-block;padding:1px 6px;border-radius:10px;font-size:0.65rem;font-weight:600;background:{tag_color}15;color:{tag_color};margin-left:4px;">{tag}</span>'
                    desc_html = f'<div style="color:#6B7280;font-size:0.8rem;margin-top:2px;line-height:1.4;">{act["desc"]}</div>' if act["desc"] else ''

                    st.markdown(
                        f'<div style="padding:5px 0 5px 12px;border-left:2px solid {bg};margin-bottom:4px;">'
                        f'<div>{time_html}{name_html}{tags_html}</div>'
                        f'{desc_html}'
                        f'</div>',
                        unsafe_allow_html=True)

            # Tips
            if day_data.get("tips"):
                tips_text = " · ".join(day_data["tips"])
                st.markdown(
                    f'<div style="background:#F9FAFB;border-radius:6px;padding:6px 10px;margin-top:6px;'
                    f'font-size:0.78rem;color:#6B7280;">💡 {tips_text}</div>',
                    unsafe_allow_html=True)


_WIKI_HEADERS = {"User-Agent": "TravelPlannerApp/1.0 (educational project; contact@example.com)"}

def fetch_wikimedia_image(query: str, size: int = 300) -> str:
    """从 Wikipedia 搜索景点照片，返回缩略图 URL。失败返回空字符串。"""
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query", "titles": query, "prop": "pageimages",
            "format": "json", "pithumbsize": size, "redirects": 1,
        }
        resp = requests.get(url, params=params, timeout=5, headers=_WIKI_HEADERS)
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            thumb = page.get("thumbnail", {}).get("source", "")
            if thumb:
                return thumb
        # fallback：搜索
        search_params = {
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": 1,
        }
        resp = requests.get(url, params=search_params, timeout=5, headers=_WIKI_HEADERS)
        results = resp.json().get("query", {}).get("search", [])
        if results:
            params["titles"] = results[0]["title"]
            resp = requests.get(url, params=params, timeout=5, headers=_WIKI_HEADERS)
            pages = resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                thumb = page.get("thumbnail", {}).get("source", "")
                if thumb:
                    return thumb
    except Exception:
        pass
    return ""


def _clean_activity_name_for_search(name: str) -> str:
    """清理活动名称，提取核心地标名用于 Wikipedia 搜索。"""
    # 去掉括号及其内容：(Exterior Views), (Accessible Paths) 等
    clean = re.sub(r'\([^)]*\)', '', name)
    # 去掉常见非地标前缀/后缀词
    skip_words = ["Budget-Friendly", "Quick", "Farewell", "Arrival", "High-Speed Train to",
                  "Vaporetto to", "Vaporetto Ride", "Exploration", "Photography",
                  "Lunch", "Dinner", "Breakfast", "Picnic", "Souvenir Hunting",
                  "Street Food", "Views", "Exterior"]
    for w in skip_words:
        clean = clean.replace(w, "")
    # 如果有 & 或 ,，取第一个部分（通常是主要地标）
    for sep in ["&", ",", " and "]:
        if sep in clean:
            clean = clean.split(sep)[0]
    clean = clean.strip(" -–—:·")
    return clean if len(clean) >= 3 else name.strip()


def batch_fetch_activity_photos(days: list[dict]) -> dict:
    """批量获取所有活动的照片。返回 {activity_name: image_url}。"""
    photos = {}
    for day in days:
        city = day.get("city", "")
        for act in day.get("activities", []):
            name = act.get("name", "")
            if not name or name in photos:
                continue
            # 清理名称提取核心地标
            clean_name = _clean_activity_name_for_search(name)
            # 尝试 1：清理后的名称（不加城市）
            img_url = fetch_wikimedia_image(clean_name)
            print(f"[Photo] query='{clean_name}' -> {'OK' if img_url else 'EMPTY'}")
            if not img_url and city:
                # 尝试 2：加城市名
                img_url = fetch_wikimedia_image(f"{clean_name} {city}")
                print(f"[Photo] +city query='{clean_name} {city}' -> {'OK' if img_url else 'EMPTY'}")
            if not img_url and clean_name != name.strip():
                # 尝试 3：原始名称第一个词（可能就是地标名）
                first_word = name.split()[0] if name.split() else ""
                if len(first_word) >= 4:
                    img_url = fetch_wikimedia_image(first_word)
                    print(f"[Photo] first-word query='{first_word}' -> {'OK' if img_url else 'EMPTY'}")
            photos[name] = img_url
    print(f"[Photo] Total: {len(photos)} entries, {sum(1 for v in photos.values() if v)} with images")
    return photos


def render_json_day_cards(days: list[dict], start_date: datetime, activity_photos: dict = None):
    """直接渲染 JSON 结构的行程数据。"""
    slot_styles = {
        "Morning": ("#FEF3C7", "#92400E", "☀️ 上午"),
        "Afternoon": ("#D1FAE5", "#065F46", "🌤️ 下午"),
        "Evening": ("#EDE9FE", "#5B21B6", "🌙 晚上"),
    }

    for day_data in days:
        day_num = day_data["day"]
        color = DAY_COLORS[(day_num - 1) % len(DAY_COLORS)]
        date_obj = start_date + timedelta(days=day_num - 1)
        date_str = date_obj.strftime("%A, %b %d")
        city = day_data.get("city", "")
        title = day_data.get("title", f"Day {day_num}")
        activities = day_data.get("activities", [])
        tips = day_data.get("tips", [])

        header = f"Day {day_num} · {date_str} — {title}"
        if city:
            header += f" ({city})"

        def _render_activity_content(act, border_color):
            """渲染单个活动的文字内容（标题、描述、tags）。"""
            cost = act.get("cost", "")
            cost_html = ""
            if cost:
                cost_color = "#059669" if cost.lower() == "free" else "#7C3AED"
                cost_html = (f'<span style="display:inline-block;padding:1px 6px;border-radius:8px;'
                             f'font-size:0.68rem;font-weight:600;background:{cost_color}12;color:{cost_color};'
                             f'margin-left:6px;">{cost}</span>')
            dur = act.get("duration", "")
            dur_html = (f'<span style="display:inline-block;padding:1px 6px;border-radius:8px;'
                        f'font-size:0.68rem;background:#F3F4F6;color:#6B7280;margin-left:4px;">{dur}</span>') if dur else ""
            desc = act.get("description", "")
            desc_html = f'<div style="font-size:0.82rem;color:#6B7280;line-height:1.5;margin-top:2px;">{desc}</div>' if desc else ""
            st.markdown(
                f'<div style="padding:6px 0 6px 10px;border-left:2px solid {border_color};margin-bottom:4px;">'
                f'<div style="display:flex;align-items:baseline;gap:6px;flex-wrap:wrap;">'
                f'<span style="font-size:0.8rem;color:#9CA3AF;min-width:52px;">{act.get("time","")}</span>'
                f'<span style="font-size:0.9rem;font-weight:600;color:#1F2937;">{act.get("name","")}</span>'
                f'{cost_html}{dur_html}'
                f'</div>'
                f'{desc_html}'
                f'</div>', unsafe_allow_html=True)

        with st.expander(header, expanded=(day_num == 1)):
            current_slot = None
            for act in activities:
                slot = act.get("slot", "Morning")
                if slot != current_slot:
                    current_slot = slot
                    bg, fg, label = slot_styles.get(slot, ("#F3F4F6", "#374151", slot))
                    st.markdown(
                        f'<div style="display:inline-block;padding:2px 10px;border-radius:6px;'
                        f'font-size:0.78rem;font-weight:600;background:{bg};color:{fg};margin:8px 0 4px;">'
                        f'{label}</div>', unsafe_allow_html=True)

                photo_url = (activity_photos or {}).get(act.get("name", ""), "")

                if photo_url:
                    img_col, text_col = st.columns([1, 3])
                    with img_col:
                        st.image(photo_url, use_container_width=True)
                    with text_col:
                        _render_activity_content(act, bg)
                else:
                    _render_activity_content(act, bg)

            if tips:
                tips_text = " · ".join(tips)
                st.markdown(
                    f'<div style="background:#F9FAFB;border-radius:6px;padding:6px 10px;margin-top:6px;'
                    f'font-size:0.78rem;color:#6B7280;line-height:1.4;">💡 {tips_text}</div>',
                    unsafe_allow_html=True)


def display_result_page():
    """行程结果页：itinerary 存在时替代输入表单。"""
    _dest = st.session_state.current_destination
    _days = st.session_state.current_days
    _bi = st.session_state.budget_info
    _itin = st.session_state.itinerary
    _prefs = st.session_state.user_preferences

    # ── a) 顶部导航栏（简化版）────────────────────
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
    logo_svg = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>'
    st.markdown(
        '<div class="top-nav">'
        '<div class="logo">'
        f'<div class="logo-icon">{logo_svg}</div>'
        '<span class="logo-text">闺蜜旅行搭子</span>'
        '</div>'
        '<div style="flex:1;"></div>'
        '</div>',
        unsafe_allow_html=True)

    # 导出 + 新建按钮
    exp_ics, exp_txt, exp_md, exp_pdf, exp_new = st.columns([1, 1, 1, 1, 1.5])
    with exp_ics:
        ics_data = generate_ics_content(_itin, st.session_state.start_date, _dest)
        st.download_button("📅 ICS", data=ics_data, file_name=f"{_dest}.ics",
                           mime="text/calendar", use_container_width=True, key="r_ics")
    with exp_txt:
        txt_data = format_itinerary_for_export(_dest, _days, _bi, _itin, "txt",
            st.session_state.start_date, st.session_state.end_date, st.session_state.packing_list)
        st.download_button("📄 TXT", data=txt_data, file_name=f"{_dest}.txt",
                           mime="text/plain", use_container_width=True, key="r_txt")
    with exp_md:
        md_data = format_itinerary_for_export(_dest, _days, _bi, _itin, "md",
            st.session_state.start_date, st.session_state.end_date, st.session_state.packing_list)
        st.download_button("📝 MD", data=md_data, file_name=f"{_dest}.md",
                           mime="text/markdown", use_container_width=True, key="r_md")
    with exp_pdf:
        try:
            pdf_data = generate_travel_guide_pdf(
                destination=_dest, num_days=_days,
                start_date=st.session_state.start_date, end_date=st.session_state.end_date,
                city_guides=st.session_state.city_guides, itinerary=_itin,
                budget_info=_bi, restaurants=st.session_state.restaurant_recommendations)
            st.download_button("📖 PDF", data=pdf_data, file_name=f"{_dest}_guide.pdf",
                               mime="application/pdf", use_container_width=True, key="r_pdf")
        except Exception:
            st.button("📖 PDF", disabled=True, use_container_width=True, key="r_pdf_err")
    with exp_new:
        if st.button("✨ 新建行程", use_container_width=True, key="r_new", type="primary"):
            for k in ['itinerary', 'budget_info', 'packing_list', 'itinerary_json', 'activity_photos']:
                st.session_state[k] = None
            st.session_state.restaurant_recommendations = []
            st.session_state.city_guides = {}
            st.session_state.photo_recognition_results = []
            for k in [k for k in st.session_state if k.startswith("map_locations_")]:
                del st.session_state[k]
            st.session_state.wizard_step = 1
            st.session_state.current_thread_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.rerun()

    # ── b) Hero 信息条 ────────────────────────────
    start_str = st.session_state.start_date.strftime("%Y.%m.%d")
    end_str = st.session_state.end_date.strftime("%Y.%m.%d")
    pref_tags = ""
    if _prefs and _prefs.get("interests"):
        pref_tags = "".join(
            f'<span style="display:inline-block;padding:3px 10px;border-radius:20px;'
            f'font-size:0.72rem;font-weight:600;background:#EDE9FE;color:#7C3AED;margin-right:4px;">{t}</span>'
            for t in _prefs["interests"][:3])
    st.markdown(
        f'<div style="background:#fff;border:1px solid #E5E7EB;border-radius:10px;'
        f'padding:14px 20px;margin-bottom:10px;display:flex;align-items:center;gap:20px;flex-wrap:wrap;">'
        f'<span style="font-size:1.3rem;font-weight:600;color:#1F2937;">{_dest}</span>'
        f'<span style="font-size:0.85rem;color:#6B7280;">{_days} 天 · {start_str} ~ {end_str}</span>'
        f'<span>{pref_tags}</span>'
        f'</div>',
        unsafe_allow_html=True)

    # ── c) Tabs ────────────────────────────────────
    tab_timeline, tab_itinerary, tab_restaurant, tab_more = st.tabs(
        ["📅 时间线", "📋 每日行程", "🍽️ 餐厅", "🧳 更多"])

    with tab_timeline:
        gantt_fig = create_itinerary_gantt(_itin, st.session_state.start_date, _days)
        if gantt_fig:
            st.plotly_chart(gantt_fig, use_container_width=True, key="result_gantt")
        else:
            st.info("暂无时间线数据")

    with tab_itinerary:
        col_text, col_map = st.columns([3, 2])
        with col_text:
            itin_json = st.session_state.get("itinerary_json")
            if itin_json and "days" in itin_json:
                # 缓存照片：检查是否有实际图片 URL（非空字符串）
                _photos_cache = st.session_state.get("activity_photos")
                _has_real_photos = _photos_cache and any(v for v in _photos_cache.values())
                if not _has_real_photos:
                    with st.spinner("🖼️ 加载景点照片..."):
                        st.session_state["activity_photos"] = batch_fetch_activity_photos(itin_json["days"])
                activity_photos = st.session_state.get("activity_photos", {})
                render_json_day_cards(itin_json["days"], st.session_state.start_date, activity_photos)
            else:
                # fallback：用旧的 parse 逻辑或原始 markdown
                days_parsed = parse_itinerary_to_days(_itin, st.session_state.start_date, _days)
                if days_parsed:
                    render_day_cards(days_parsed)
                else:
                    st.markdown(add_dates_to_itinerary(_itin, st.session_state.start_date, _days))

        with col_map:
            if gemini_api_key:
                llm_map = ChatGoogleGenerativeAI(model=HELPER_MODEL, google_api_key=gemini_api_key, temperature=0.3)
                display_itinerary_map(_itin, _dest, _days, st.session_state.start_date, llm_map, "result")
            else:
                st.info("需要 API key 才能显示地图")

    with tab_restaurant:
        if st.session_state.restaurant_recommendations:
            display_restaurant_cards(st.session_state.restaurant_recommendations)
        else:
            st.info("暂无餐厅推荐")

    with tab_more:
        # 打包清单
        if st.session_state.packing_list:
            with st.expander("🧳 打包清单", expanded=False):
                st.markdown(st.session_state.packing_list)
                st.download_button("📥 下载打包清单", data=st.session_state.packing_list,
                                   file_name=f"{_dest}_packing.md", mime="text/markdown",
                                   use_container_width=True, key="r_packing")

        # 语音导览
        if gemini_api_key:
            with st.expander("🎧 语音导览", expanded=False):
                display_audio_tour_feature(
                    itinerary=_itin, destination=_dest, num_days=_days,
                    start_date=st.session_state.start_date, gemini_api_key=gemini_api_key,
                    stepfun_api_key=stepfun_api_key)

        # 照片识别景点
        if gemini_api_key and _bi:
            with st.expander("📸 照片识别景点", expanded=False):
                display_photo_landmark_feature(
                    current_itinerary=_itin, destination=_dest, num_days=_days,
                    budget=_bi['total'], start_date=st.session_state.start_date,
                    gemini_api_key=gemini_api_key)

        # 小红书内容校验
        if gemini_api_key:
            with st.expander("🔍 小红书内容校验", expanded=False):
                # _serp 参数保留兼容签名，实际搜索用 tavily_client
                display_fact_check_feature(None, gemini_api_key)

    # ── 底部固定修改行程入口（所有 tabs 之后）──
    feedback = st.chat_input('哪里不满意？比如 "Day 2 太赶了" 或 "加一些咖啡馆"')
    if feedback:
        try:
            with st.spinner("✍️ 正在修改行程..."):
                app = create_travel_agent_graph(gemini_api_key, serp_api_key)
                revision_state = {
                    "messages": [],
                    "destination": _dest,
                    "num_days": _days,
                    "budget": _bi['total'] if _bi else 1000,
                    "final_itinerary": _itin,
                    "user_feedback": feedback,
                    "feedback_analysis": {},
                    "iteration": 0,
                    "packing_list": st.session_state.packing_list or "",
                    "weather_data": st.session_state.weather_data or {},
                    "user_preferences": _prefs,
                    "restaurant_recommendations": st.session_state.restaurant_recommendations or [],
                    "city_guides": st.session_state.city_guides or {},
                    "itinerary_json": st.session_state.get("itinerary_json"),
                }
                rev_id = f"{st.session_state.current_thread_id}_rev_{datetime.now().strftime('%H%M%S')}"
                config = {"configurable": {"thread_id": rev_id}}
                result = app.invoke(revision_state, config=config)
                st.session_state.itinerary = result["final_itinerary"]
                st.session_state.itinerary_json = result.get("itinerary_json")
                st.session_state.pop("activity_photos", None)
                for k in [k for k in st.session_state if k.startswith("map_locations_")]:
                    del st.session_state[k]
                llm_h = ChatGoogleGenerativeAI(model=HELPER_MODEL, google_api_key=gemini_api_key, temperature=0.3)
                acts = extract_activities_from_itinerary(result["final_itinerary"], llm_h)
                st.session_state.packing_list = generate_packing_list(_dest, _days, st.session_state.weather_data or {}, acts)
                st.success("✅ 行程已修改！")
                st.rerun()
        except Exception as e:
            st.error(f"修改失败: {str(e)}")


# ==================== Main Interface ====================
if gemini_api_key and (tavily_api_key or serp_api_key):
    if st.session_state.itinerary:
        # 有行程 → 结果页
        display_result_page()
    else:
        # 无行程 → 输入流程
        _mode = st.session_state.home_mode
        _step = st.session_state.wizard_step if _mode == "plan" else 1

        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        render_nav_bar(_step)

        # 模式切换：两个按钮
        col_plan, col_vibe = st.columns(2)
        with col_plan:
            if st.button("直接规划", type="primary" if _mode == "plan" else "secondary",
                          use_container_width=True, key="mode_plan"):
                st.session_state.home_mode = "plan"
                st.rerun()
        with col_vibe:
            if st.button("Vibe Starting", type="primary" if _mode == "vibe" else "secondary",
                          use_container_width=True, key="mode_vibe"):
                st.session_state.home_mode = "vibe"
                st.rerun()

        # ================================================================
        #  MODE A: 直接规划（结构化表单 Stepper）
        # ================================================================
        if st.session_state.home_mode == "plan":
            step = st.session_state.wizard_step

            # ──────────── STEP 1: 去哪玩 ────────────
            if step == 1:
                left_col, right_col = st.columns([1, 1], gap="large")

                with left_col:
                    st.markdown('<p style="font-size:20px;font-weight:500;color:#111827;margin:0 0 2px;">规划你的旅程</p>'
                                '<p style="font-size:13px;color:#9CA3AF;margin:0 0 16px;">填写基本信息，AI 帮你生成专属攻略</p>',
                                unsafe_allow_html=True)

                    destination = st.text_input("目的地", placeholder="例如：Tokyo、Paris、Bangkok...",
                                                value=st.session_state.current_destination)

                    c_days, c_date = st.columns(2)
                    with c_days:
                        default_days = st.session_state.get('template_days', st.session_state.current_days)
                        num_days = st.number_input("天数", min_value=1, max_value=30, value=default_days)
                        if st.session_state.template_applied:
                            st.session_state.template_applied = False
                    with c_date:
                        _min_date = min(datetime.today(), st.session_state.start_date)
                        start_date = st.date_input("出发日期", value=st.session_state.start_date,
                                                   min_value=_min_date)
                        start_datetime = datetime.combine(start_date, datetime.min.time())
                        end_datetime = start_datetime + timedelta(days=num_days - 1)
                        st.session_state.start_date = start_datetime
                        st.session_state.end_date = end_datetime

                    # ── 天气概览条 ──
                    if destination:
                        render_weather_bar(destination, start_datetime, num_days,
                                           weather_api_key or None)

                    if st.button("下一步", type="primary", use_container_width=True, key="step1_next"):
                        if not destination:
                            st.error("请先输入目的地")
                        else:
                            st.session_state.current_destination = destination
                            st.session_state.current_days = num_days
                            st.session_state.wizard_step = 2
                            st.rerun()

                with right_col:
                    # 右侧视觉区
                    _dest_display = destination if destination else "你的下一站"
                    _days_display = num_days if destination else ""
                    _days_text = f"{_days_display} 天探索之旅" if _days_display else "等你来规划"

                    travel_svg = '<svg width="160" height="120" viewBox="0 0 180 140" fill="none"><rect x="55" y="60" width="70" height="55" rx="8" fill="#DDD6FE" stroke="#8B5CF6" stroke-width="1.5"/><rect x="70" y="52" width="40" height="12" rx="4" fill="none" stroke="#8B5CF6" stroke-width="1.5"/><line x1="55" y1="78" x2="125" y2="78" stroke="#8B5CF6" stroke-width="1"/><circle cx="72" cy="120" r="5" fill="#C4B5FD" stroke="#8B5CF6" stroke-width="1.5"/><circle cx="108" cy="120" r="5" fill="#C4B5FD" stroke="#8B5CF6" stroke-width="1.5"/><g transform="translate(110,15) rotate(15)"><path d="M0 12 L25 8 L40 0 L42 4 L28 10 L50 14 L50 18 L28 16 L25 16 L42 28 L38 30 L20 20 L8 18 Z" fill="#FBCFE8" stroke="#EC4899" stroke-width="1"/></g><g transform="translate(25,25)"><path d="M10 0 C4.5 0 0 4.5 0 10 C0 17 10 28 10 28 C10 28 20 17 20 10 C20 4.5 15.5 0 10 0Z" fill="#FDE68A" stroke="#D97706" stroke-width="1.2"/><circle cx="10" cy="10" r="4" fill="#fff"/></g></svg>'

                    tags_html = ""
                    if destination:
                        tags_html = (
                            '<div class="tag-row">'
                            '<span class="tag" style="background:rgba(139,92,246,0.1);color:#7C3AED;">AI 智能规划</span>'
                            '<span class="tag" style="background:rgba(236,72,153,0.1);color:#DB2777;">个性化推荐</span>'
                            '<span class="tag" style="background:rgba(217,119,6,0.1);color:#B45309;">实时天气</span>'
                            '</div>'
                        )

                    st.markdown(
                        f'<div class="visual-panel">{travel_svg}'
                        f'<div class="dest-name">{_dest_display}</div>'
                        f'<div class="dest-sub">{_days_text}</div>'
                        f'{tags_html}</div>',
                        unsafe_allow_html=True)

            # ──────────── STEP 2: 偏好 ────────────
            elif step == 2:
                pass  # nav bar handles step indicator

                user_preferences = collect_user_preferences()
                if user_preferences:
                    st.session_state.user_preferences = user_preferences

                col_back, _, col_next = st.columns([1, 3, 1])
                with col_back:
                    if st.button("← 上一步", use_container_width=True, key="step2_back"):
                        st.session_state.wizard_step = 1
                        st.rerun()
                with col_next:
                    if st.button("下一步 →", type="primary", use_container_width=True, key="step2_next"):
                        st.session_state.wizard_step = 3
                        st.rerun()

            # ──────────── STEP 3: 确认 & 生成 ────────────
            elif step == 3:
                pass  # nav bar handles step indicator

                dest = st.session_state.current_destination
                days = st.session_state.current_days
                prefs = st.session_state.user_preferences

                # 预算：UI 隐藏后仍需传给 agent，使用默认档
                budget_level = st.session_state.wizard_budget_level
                custom_budget = st.session_state.wizard_custom_budget
                currency = st.session_state.wizard_currency
                if custom_budget:
                    budget_info = {
                        "total": custom_budget, "daily": custom_budget / days,
                        "breakdown": {
                            "accommodation": custom_budget * 0.35, "food": custom_budget * 0.25,
                            "transportation": custom_budget * 0.15, "activities": custom_budget * 0.20,
                            "miscellaneous": custom_budget * 0.05,
                        }
                    }
                else:
                    budget_info = estimate_budget(dest, days, budget_level)

                # ── 总览卡片 ──
                pref_tags_html = ""
                if prefs:
                    tags = []
                    if prefs.get("interests"):
                        tags.extend(prefs["interests"][:3])
                    if prefs.get("traveling_with"):
                        tags.extend(prefs["traveling_with"][:2])
                    if prefs.get("travel_pace"):
                        tags.append(prefs["travel_pace"].split("(")[0].strip())
                    if prefs.get("dietary_restrictions"):
                        tags.append(prefs["dietary_restrictions"])
                    pref_tags_html = '<div class="pref-tags">' + "".join(
                        f'<span class="pref-tag">{t}</span>' for t in tags) + '</div>'

                start_str = st.session_state.start_date.strftime("%Y.%m.%d")
                end_str = st.session_state.end_date.strftime("%Y.%m.%d")

                st.markdown(f"""
                <div class="summary-card">
                    <div class="row">
                        <span class="label">目的地</span>
                        <span class="value" style="font-size:1.15rem;">🌍 {dest}</span>
                    </div>
                    <div class="row">
                        <span class="label">行程</span>
                        <span class="value">📅 {days} 天 · {start_str} ~ {end_str}</span>
                    </div>
                    <div class="row">
                        <span class="label">偏好</span>
                        <span class="value">{pref_tags_html if pref_tags_html else '<span style="color:var(--grey-500);">未设置</span>'}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_back, _, col_gen = st.columns([1, 1, 2])
                with col_back:
                    if st.button("✏️ 返回修改", use_container_width=True, key="step3_back"):
                        st.session_state.wizard_step = 1
                        st.rerun()
                with col_gen:
                    generate_button = st.button("🚀 生成我的攻略", type="primary", use_container_width=True, key="step3_gen")

                if generate_button:
                    try:
                        final_budget = budget_info['total']
                        st.session_state.budget_info = budget_info
                        st.session_state.current_thread_id = datetime.now().strftime("%Y%m%d_%H%M%S")

                        keys_to_remove = [k for k in st.session_state if k.startswith("map_locations_")]
                        for k in keys_to_remove:
                            del st.session_state[k]

                        app = create_travel_agent_graph(gemini_api_key, serp_api_key)
                        initial_state = {
                            "messages": [], "destination": dest, "num_days": days,
                            "budget": final_budget, "final_itinerary": "", "user_feedback": "",
                            "feedback_analysis": {}, "iteration": 0, "packing_list": "",
                            "weather_data": st.session_state.weather_data or {},
                            "user_preferences": st.session_state.user_preferences,
                            "restaurant_recommendations": [], "city_guides": {}, "itinerary_json": None,
                        }

                        status_container = st.empty()
                        status_container.info("🔍 研究员正在全网搜集信息...")

                        config = {"configurable": {"thread_id": st.session_state.current_thread_id}}
                        result = app.invoke(initial_state, config=config)

                        status_container.info("🧳 正在生成打包清单...")
                        st.session_state.itinerary = result["final_itinerary"]
                        st.session_state.itinerary_json = result.get("itinerary_json")
                        st.session_state.restaurant_recommendations = result.get("restaurant_recommendations", [])
                        st.session_state.city_guides = result.get("city_guides", {})

                        llm_helper = ChatGoogleGenerativeAI(model=HELPER_MODEL, google_api_key=gemini_api_key, temperature=0.3)
                        activities = extract_activities_from_itinerary(result["final_itinerary"], llm_helper)
                        packing_list = generate_packing_list(dest, days, st.session_state.weather_data or {}, activities)
                        st.session_state.packing_list = packing_list

                        weather_key = weather_api_key or None
                        weather_data = get_weather_forecast(dest, weather_key)
                        if weather_data:
                            st.session_state.weather_data = weather_data

                        save_to_history(dest, days, budget_info, result["final_itinerary"],
                                        st.session_state.start_date, st.session_state.end_date, packing_list)

                        status_container.empty()
                        st.success("✅ 你的专属攻略生成好啦！往下看~")
                        st.rerun()

                    except Exception as e:
                        st.error(f"生成失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

        # ================================================================
        #  MODE B: Vibe Starting（对话式）
        # ================================================================
        elif st.session_state.home_mode == "vibe":
            if not st.session_state.vibe_messages:
                st.markdown("""
                <div style="text-align:center; padding:1rem 1rem 0.5rem; color:var(--grey-500);">
                    <div style="font-size:0.95rem; font-weight:600; color:var(--navy); margin-bottom:2px;">
                        告诉我你的旅行想法
                    </div>
                    <div style="font-size:0.82rem; line-height:1.5;">
                        不用很具体 — "想去海边放空" "想和闺蜜拍好看的照片" "预算3000想玩一周" 都行
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 显示对话历史
            for msg in st.session_state.vibe_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # 对话输入
            vibe_input = st.chat_input("随便说说你的旅行想法...")

            if vibe_input:
                st.session_state.vibe_messages.append({"role": "user", "content": vibe_input})

                vibe_llm = ChatGoogleGenerativeAI(
                    model=HELPER_MODEL, google_api_key=gemini_api_key, temperature=0.8)
                vibe_system = """你是一个友好的旅行规划助手。用户会用模糊的方式描述旅行想法。
你的任务是：
1. 理解用户的 vibe（氛围、心情、偏好）
2. 推荐 2-3 个具体目的地，简短说明为什么适合
3. 询问用户想选哪个，或者继续聊
4. 保持回复简洁（100字以内），像闺蜜聊天的语气
5. 当用户确定了目的地和大致天数后，告诉他们可以点击"直接规划"开始生成详细行程"""

                messages = [SystemMessage(content=vibe_system)]
                for msg in st.session_state.vibe_messages[-6:]:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))

                with st.spinner(""):
                    response = vibe_llm.invoke(messages)
                    reply = response.content
                    st.session_state.vibe_messages.append({"role": "assistant", "content": reply})
                    st.rerun()

else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding:3rem 2rem; background:linear-gradient(135deg, #FFF5F7, #F3F0FF);
                    border-radius:20px; border:1px solid #F1F5F9; margin-top:1rem;">
            <div style="font-size:3rem; margin-bottom:8px;">🔑</div>
            <div style="font-size:1.2rem; font-weight:700; color:#1E293B; margin-bottom:8px;">先配置 API Key 再开始~</div>
            <div style="font-size:0.9rem; color:#64748B;">在左边侧边栏输入你的 API Key 就可以用啦</div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("💡 API Key 怎么拿？"):
            st.markdown("""
**Google Gemini API（免费）：**
1. 打开 [Google AI Studio](https://makersuite.google.com/app/apikey)
2. 登录 Google 账号
3. 创建 API Key（有免费额度）

**SerpAPI（免费100次/月）：**
1. 打开 [SerpAPI](https://serpapi.com)
2. 注册账号
3. 拿到免费 API Key
""")

# (旧的 Current Itinerary 和 Xiaohongshu 区块已迁移到 display_result_page())